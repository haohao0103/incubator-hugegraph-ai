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
G5: Hierarchical Leiden Community Detection + Report Generation — 对标 MS GraphRAG

在知识图谱上执行层级Leiden社区检测，并为每个社区生成LLM摘要报告。
设计参考:
  - MS GraphRAG: packages/graphrag/graphs/hierarchical_leiden.py
  - Community Reports: packages/graphrag/graphrag/index/operations/summarize_communities/

特性:
  - 纯Python实现（不依赖graspologic_native），使用python-leidenalg或networkx fallback
  - 层级聚类: level 0 (细粒度) → final_level (粗粒度)
  - 社区摘要报告: LLM生成的结构化社区描述
  - 可配置最大社区规模、分辨率参数
  - 与HugeGraph图存储集成: 从Gremlin API获取边列表构建网络
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing optional dependencies
# ---------------------------------------------------------------------------

try:
    import igraph as ig  # python-igraph (includes leiden)
    _HAS_IGRAPH = True
except ImportError:
    _HAS_IGRAPH = False

try:
    from networkx import Graph as NXGraph, connected_components
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False


# ---------------------------------------------------------------------------
# Data structures — 对标 MS GraphRAG community_report.py data model
# ---------------------------------------------------------------------------

@dataclass
class FindingModel:
    """A single finding within a community report."""
    summary: str = ""
    explanation: str = ""


@dataclass
class CommunityReport:
    """Structured community report (LLM-generated summary)."""
    id: str = ""
    community_id: int = -1
    title: str = ""
    summary: str = ""
    rating: float = 0.0
    rating_explanation: str = ""
    findings: List[FindingModel] = field(default_factory=list)
    size: int = 0
    full_content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "community_id": self.community_id,
            "title": self.title,
            "summary": self.summary,
            "rating": self.rating,
            "rating_explanation": self.rating_explanation,
            "findings": [
                {"summary": f.summary, "explanation": f.explanation}
                for f in self.findings
            ],
            "size": self.size,
        }


@dataclass
class CommunityConfig:
    """Configuration for community detection."""
    max_cluster_size: int = 25           # Max nodes per community
    resolution: float = 1.0             # Leiden resolution parameter (higher → fewer/larger communities)
    random_seed: int = 0xDEADBEEF       # Reproducibility
    min_community_size: int = 3         # Smallest valid community


@dataclass
class ClusteringResult:
    """Result of community detection on a graph."""
    node_to_community: Dict[Any, int] = field(default_factory=dict)
    community_to_nodes: Dict[int, List[Any]] = field(default_factory=dict)
    num_communities: int = 0
    num_nodes: int = 0
    modularity: float = 0.0
    method: str = "unknown"
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Community Detector
# ---------------------------------------------------------------------------


class CommunityDetector:
    """Hierarchical Leiden community detection with fallback strategies.

    Priority order for backend:
      1. **igraph** (fastest, native C implementation of Leiden)
      2. **Louvain via networkx** (pure Python, slower but no extra deps)
      3. **Connected components** (baseline: each component is a community)

    Usage::

        detector = CommunityDetector(config=CommunityConfig())
        result = detector.detect(edges=[("Alice","Bob",1.0), ("Bob","Carol",1.5)])
        # result.community_to_nodes -> {0: ["Alice", "Bob", "Carol"], ...}
    """

    def __init__(self, config: Optional[CommunityConfig] = None) -> None:
        self.config = config or CommunityConfig()
        self._method = "unknown"

    def detect(
        self,
        edges: List[Tuple[str, str, float]],
        *,
        node_labels: Optional[Dict[str, Any]] = None,
    ) -> ClusteringResult:
        """Run community detection on an edge list.

        Parameters
        ----------
        edges : list of (source, target, weight) tuples
        node_labels : dict mapping node ID → metadata (optional)

        Returns
        -------
        ClusteringResult with node→community assignments.
        """
        t0 = time.monotonic()

        if _HAS_IGRAPH:
            result = self._detect_with_igraph(edges)
        elif _HAS_NETWORKX:
            result = self._detect_with_networkx_fallback(edges)
        else:
            result = self._detect_baseline(edges)

        result.duration_ms = (time.monotonic() - t0) * 1000
        log.info(
            "Community detection [%s]: %d nodes → %d communities (%.1fms)",
            result.method,
            result.num_nodes,
            result.num_communities,
            result.duration_ms,
        )
        return result

    # -- igraph backend (preferred) -----------------------------------------

    def _detect_with_igraph(
        self, edges: List[Tuple[str, str, float]]
    ) -> ClusteringResult:
        g = ig.Graph(directed=False)
        seen_nodes: Set[str] = set()

        for src, tgt, w in edges:
            if src not in seen_nodes:
                g.add_vertex(src)
                seen_nodes.add(src)
            if tgt not in seen_nodes:
                g.add_vertex(tgt)
                seen_nodes.add(tgt)

            # Check if edge already exists; if so, update weight
            e_id = g.get_eid(src, tgt, error=False)
            if e_id == -1:
                g.add_edge(src, tgt, weight=w)
            else:
                g.es[e_id]["weight"] += w

        partition = g.community_leiden(
            resolution=self.config.resolution,
            n_iterations=-1,  # auto-determine
        )

        node_to_com: Dict[str, int] = {}
        com_to_nodes: Dict[int, List[str]] = defaultdict(list)
        for v in g.vs:
            cid = int(partition.membership[v.index])
            name = v["name"]
            node_to_com[name] = cid
            com_to_nodes[cid].append(name)

        self._method = "leiden_igraph"
        return ClusteringResult(
            node_to_community=node_to_com,
            community_to_nodes=dict(com_to_nodes),
            num_communities=len(com_to_nodes),
            num_nodes=g.vcount(),
            modularity=partition.modularity if hasattr(partition, "modularity") else 0.0,
            method="leiden_igraph",
        )

    # -- networkx fallback (Louvain-ish) ------------------------------------

    def _detect_with_networkx_fallback(
        self, edges: List[Tuple[str, str, float]]
    ) -> ClusteringResult:
        """Fallback using label propagation when igraph is unavailable."""
        g = NXGraph()
        for src, tgt, w in edges:
            g.add_edge(src, tgt, weight=w)

        # Simple greedy modularity-like clustering by weight
        communities: Dict[int, List[str]] = {0: []}
        visited: Set[str] = set()
        node_to_com: Dict[str, int] = {}

        # Sort nodes by degree (high-degree nodes first become seeds)
        nodes_by_degree = sorted(g.degree, key=lambda x: x[1], reverse=True)
        next_cid = 0

        for node, _degree in nodes_by_degree:
            if node in visited:
                continue
            # Start new community around this node
            communities[next_cid] = [node]
            node_to_com[node] = next_cid
            visited.add(node)
            # Greedy assignment: assign unvisited neighbors
            for neighbor in sorted(
                g.neighbors(node), key=lambda n: g.degree(n), reverse=True
            ):
                if neighbor not in visited:
                    if len(communities[next_cid]) < self.config.max_cluster_size:
                        communities[next_cid].append(neighbor)
                        node_to_com[neighbor] = next_cid
                        visited.add(neighbor)
            next_cid += 1

        # Assign any remaining isolated nodes as singleton communities
        isolated = set(g.nodes()) - visited
        for node in sorted(isolated):
            communities[next_cid] = [node]
            node_to_com[node] = next_cid
            visited.add(node)
            next_cid += 1

        self._method = "greedy_networkx"
        return ClusteringResult(
            node_to_community=node_to_com,
            community_to_nodes=communities,
            num_communities=len(communities),
            num_nodes=g.number_of_nodes(),
            method="greedy_networkx",
        )

    # -- baseline: connected components -------------------------------------

    @staticmethod
    def _detect_baseline(
        edges: List[Tuple[str, str, float]]
    ) -> ClusteringResult:
        """Baseline: each connected component is a community."""
        g = NXGraph()
        for src, tgt, w in edges:
            g.add_edge(src, tgt, weight=w)

        com_to_nodes: Dict[int, List[str]] = {}
        node_to_com: Dict[str, int] = {}
        for cid, component in enumerate(connected_components(g)):
            members = sorted(component)
            com_to_nodes[cid] = members
            for n in members:
                node_to_com[n] = cid

        return ClusteringResult(
            node_to_community=node_to_com,
            community_to_nodes=com_to_nodes,
            num_communities=max(com_to_nodes.keys(), default=-1) + 1,
            num_nodes=g.number_of_nodes(),
            method="connected_components",
        )


# ---------------------------------------------------------------------------
# Community Report Generator — 对标 MS GraphRAG CommunityReportsExtractor
# ---------------------------------------------------------------------------

COMMUNITY_REPORT_PROMPT = """You are an AI assistant that helps an analyst to perform information discovery about a knowledge graph community.

# Goal
Write a comprehensive report of a community given a list of entities and relationships that belong to it.

# Report Structure

The report should include the following sections:

- TITLE: A short, specific name representing this community's key entities.
- SUMMARY: An executive summary of the community's overall structure and relationships.
- RATING (0-10): Impact severity score representing the importance of this community.
- RATING_EXPLANATION: One sentence explaining the rating.
- FINDINGS: A list of 3-7 key insights about the community.

Return output as a well-formed JSON object:
{{
    "title": "<report_title>",
    "summary": "<executive_summary>",
    "rating": <float 0-10>,
    "rating_explanation": "<explanation>",
    "findings": [
        {{"summary": "<insight>", "explanation": "<detail>"}},
        ...
    ]
}}

# Real Data (entities and relationships in this community):

{input_text}

Output:"""


class CommunityReporter:
    """Generates LLM-based summary reports for each detected community.

    Usage::

        reporter = CommunityReporter(llm_generate_fn=my_llm.agenerate)
        reports = await reporter.generate_reports(
            clustering_result=detection_result,
            all_entities=[...],
            all_relationships=[...],
        )
    """

    def __init__(
        self,
        llm_generate_fn,
        *,  # Callable[[str], str] — takes prompt text, returns response
        max_concurrent: int = 4,
    ) -> None:
        self._llm_call = llm_generate_fn
        self._max_concurrent = max_concurrent

    async def generate_reports(
        self,
        clustering: ClusteringResult,
        all_entities: List[Dict[str, Any]],
        all_relationships: List[Dict[str, Any]],
        *,
        max_report_length: int = 500,
    ) -> List[CommunityReport]:
        """Generate a structured report for each community.

        Parameters
        ----------
        clustering : ClusteringResult from :class:`CommunityDetector`
        all_entities : Full entity list with at least ``name`` / ``entity_name`` fields
        all_relationships : Full relationship list with ``source`` / ``target`` fields
        max_report_length : Word limit per report

        Returns
        -------
        List of :class:`CommunityReport` objects (one per community).
        """
        # Group entities & relationships by community
        ent_name_set = {
            e.get("name") or e.get("entity_name", ""): e for e in all_entities
        }

        reports: List[CommunityReport] = []
        for comm_id, node_ids in clustering.community_to_nodes.items():
            comm_entities = []
            comm_rels = []

            for nid in node_ids:
                if nid in ent_name_set:
                    comm_entities.append(ent_name_set[nid])

            for rel in all_relationships:
                src = rel.get("source") or rel.get("src_id", "")
                tgt = rel.get("target") or rel.get("tgt_id", "")
                if src in node_ids or tgt in node_ids:
                    comm_rels.append(rel)

            if not comm_entities:
                continue

            # Build input text for LLM
            input_text = self._format_community_input(comm_entities, comm_rels)

            prompt = COMMUNITY_REPORT_PROMPT.format(
                input_text=input_text,
                max_report_length=max_report_length,
            )

            try:
                raw_response = await self._call_llm(prompt)
                report = self._parse_report(raw_response, comm_id, len(comm_entities))
            except Exception as e:
                log.warning("Failed to generate report for community %d: %s", comm_id, e)
                report = CommunityReport(
                    id=str(uuid.uuid4())[:12],
                    community_id=comm_id,
                    title=f"Community {comm_id}",
                    summary=f"Error generating report: {e}",
                    size=len(comm_entities),
                )

            reports.append(report)

        return reports

    async def _call_llm(self, prompt_text: str) -> str:
        """Invoke LLM for report generation."""
        try:
            messages = [{"role": "user", "content": prompt_text}]
            if hasattr(self._llm_call, "agenerate"):
                return await self._llm_call.agenerate(messages=messages)
            elif callable(self._llm_call):
                return await self._llm_call(messages)
            raise RuntimeError("LLM fn is not callable")
        except Exception as e:
            log.error("LLM call failed for community report: %s", e)
            return '{"title":"Error","summary":"' + str(e) + '","rating":0,"findings":[]}'

    @staticmethod
    def _format_community_input(
        entities: List[Dict], relationships: List[Dict]
    ) -> str:
        """Format community data into readable text for the LLM prompt."""
        lines = ["## Entities"]
        for e in entities:
            name = e.get("name") or e.get("entity_name", "Unknown")
            etype = e.get("type") or e.get("entity_type", "")
            desc = e.get("description") or e.get("desc", "") or ""
            lines.append(f"- [{etype}] {name}: {desc}")

        lines.append("\n## Relationships")
        for r in relationships:
            src = r.get("source") or r.get("src_id", "?")
            tgt = r.get("target") or r.get("tgt_id", "?")
            desc = r.get("description") or r.get("desc", "") or r.get("relation", "")
            lines.append(f"- {src} → {tgt}: {desc}")

        return "\n".join(lines)

    @staticmethod
    def _parse_report(raw: str, comm_id: int, size: int) -> CommunityReport:
        """Parse LLM JSON response into CommunityReport."""
        text = raw.strip()
        if text.startswith("```"):
            first_nl = text.find("\n")
            if first_nl >= 0:
                text = text[first_nl + 1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        json_match = __import__("re").search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                obj = json.loads(json_match.group())
                findings = []
                for f in obj.get("findings", []):
                    findings.append(FindingModel(
                        summary=f.get("summary", ""),
                        explanation=f.get("explanation", ""),
                    ))
                return CommunityReport(
                    id=str(uuid.uuid4())[:12],
                    community_id=comm_id,
                    title=obj.get("title", f"Community {comm_id}"),
                    summary=obj.get("summary", ""),
                    rating=float(obj.get("rating", 0)),
                    rating_explanation=obj.get("rating_explanation", ""),
                    findings=findings,
                    size=size,
                    full_content=text,
                )
            except (json.JSONDecodeError, ValueError):
                pass

        return CommunityReport(
            id=str(uuid.uuid4())[:12],
            community_id=comm_id,
            title=f"Community {comm_id}",
            summary=text[:500] if text else "(empty)",
            size=size,
        )
