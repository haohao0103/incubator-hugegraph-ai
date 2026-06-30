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

"""Community Summary Generator — bridges CommunityDetect → GlobalSearchRetriever.

Generates LLM-based community summaries from detected communities and stores
them as CommunityReport objects that can be used by GlobalSearchRetriever.

This completes the P0-5 gap: we have community detection (community_detect.py)
and global search retrieval (global_retriever.py), but the missing link is
generating community summaries and storing them for retrieval.

Design references:
    - MS-GraphRAG: graphrag/index/operations/create_final_community_reports.py
      (LLM generates title + summary + findings for each community)
    - LightRAG: community-aware retrieval via hl_keywords → relationships VDB
    - Fast-GraphRAG: no community summaries (uses PPR propagation instead)
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from hugegraph_llm.utils.log import log

# ── Data structures ───────────────────────────────────────────────


@dataclass
class CommunityFinding:
    """A single finding within a community report."""
    summary: str = ""
    explanation: str = ""


@dataclass
class CommunityReport:
    """A generated community report (对标 MS GraphRAG CommunityReport)."""
    id: str = ""                           # Community ID
    title: str = ""                        # LLM-generated title
    summary: str = ""                       # LLM-generated summary
    findings: List[CommunityFinding] = field(default_factory=list)
    level: int = 0                          # Community hierarchy level
    rank: float = 0.0                       # Relevance rank for retrieval
    embedding: Optional[List[float]] = None # Summary embedding for VDB search
    entity_count: int = 0                   # Number of entities in this community
    edge_count: int = 0                     # Number of edges in this community

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "findings": [{"summary": f.summary, "explanation": f.explanation} for f in self.findings],
            "level": self.level,
            "rank": self.rank,
            "entity_count": self.entity_count,
            "edge_count": self.edge_count,
        }

    def full_content(self) -> str:
        """Full text content for embedding search."""
        parts = [f"# {self.title}", self.summary]
        for f in self.findings:
            parts.append(f"- {f.summary}: {f.explanation}")
        return "\n".join(parts)


# ── LLM Prompt template ──────────────────────────────────────────

COMMUNITY_SUMMARY_PROMPT = """You are an expert analyst tasked with generating a comprehensive community report for a knowledge graph community.

---Community Data---
Community ID: {community_id}
Entities in this community:
{entity_list}

Relationships in this community:
{relation_list}

---Task---
Generate a structured report for this community with the following sections:

1. **Title**: A concise, descriptive title for this community (max 10 words).
2. **Summary**: A comprehensive summary of what this community represents, its key themes, and main entities (2-4 sentences).
3. **Findings**: 3-5 key findings or insights from the community structure, each with a summary and explanation.

---Output Format---
Return a valid JSON object with exactly these keys:
{{"title": "...", "summary": "...", "findings": [{{"summary": "...", "explanation": "..."}}]}}

---Output---
"""


# ── Heuristic fallback (no LLM) ──────────────────────────────────

HEURISTIC_COMMUNITY_TITLE_PROMPT = """Auto-generate a community title from entity names."""


@dataclass
class CommunitySummaryConfig:
    """Configuration for community summary generation."""
    max_entities_per_report: int = 50      # Max entities to include in LLM prompt
    max_relations_per_report: int = 30     # Max relations to include in LLM prompt
    max_findings: int = 5                  # Max findings per report
    llm_max_retries: int = 1               # Max retries for LLM call
    fallback_to_heuristic: bool = True     # Use heuristic if LLM fails
    generate_embeddings: bool = True       # Generate embeddings for reports


class CommunitySummaryGenerator:
    """Generate community summaries from detected communities.

    Bridges CommunityDetect (community assignments) → GlobalSearchRetriever
    (community reports for retrieval).

    Two generation modes:
    1. **LLM mode**: Uses LLM to generate title + summary + findings
       (对标 MS GraphRAG create_final_community_reports).
    2. **Heuristic mode**: Uses entity/relation statistics when LLM unavailable
       (generates descriptive title from top entities, statistical summary).

    Usage::

        generator = CommunitySummaryGenerator(llm=my_llm, embedding_fn=my_embed)
        reports = generator.generate(
            communities={0: ["Entity_A", "Entity_B"], 1: ["Entity_C"]},
            entity_texts={"Entity_A": "description...", ...},
            relations=[{"source": "Entity_A", "target": "Entity_B", "desc": "..."}],
        )
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        config: Optional[CommunitySummaryConfig] = None,
    ) -> None:
        self._llm = llm
        self._embedding_fn = embedding_fn
        self.config = config or CommunitySummaryConfig()

    def generate(
        self,
        communities: Dict[int, List[str]],    # community_id → [entity_names]
        entity_texts: Dict[str, str],          # entity_name → description
        relations: Optional[List[Dict[str, Any]]] = None,
        edge_texts: Optional[Dict[str, str]] = None,
    ) -> List[CommunityReport]:
        """Generate community reports from detected communities.

        Args:
            communities: Dict mapping community ID to list of entity names.
            entity_texts: Dict mapping entity name to text description.
            relations: Optional list of relation dicts with source/target/description.
            edge_texts: Optional dict mapping edge_id to description text.

        Returns:
            List of CommunityReport objects, ready for GlobalSearchRetriever.
        """
        t0 = time.perf_counter()
        reports: List[CommunityReport] = []

        for comm_id, entity_names in communities.items():
            if not entity_names:
                continue

            # Filter relations belonging to this community
            comm_relations = self._filter_community_relations(
                entity_names, relations or []
            )

            report = self._generate_single_report(
                comm_id, entity_names, entity_texts, comm_relations, edge_texts,
            )
            reports.append(report)

        # Sort by entity_count descending (larger communities first)
        reports.sort(key=lambda r: r.entity_count, reverse=True)

        elapsed = time.perf_counter() - t0
        log.info(f"[CommunitySummary] Generated {len(reports)} reports in {elapsed:.2f}s")
        return reports

    def _generate_single_report(
        self,
        comm_id: int,
        entity_names: List[str],
        entity_texts: Dict[str, str],
        comm_relations: List[Dict[str, Any]],
        edge_texts: Optional[Dict[str, str]],
    ) -> CommunityReport:
        """Generate a single community report.

        Tries LLM first, falls back to heuristic if LLM unavailable.
        """
        # Truncate entities and relations for LLM prompt
        entities_subset = entity_names[:self.config.max_entities_per_report]
        relations_subset = comm_relations[:self.config.max_relations_per_report]

        # Try LLM generation
        if self._llm:
            for attempt in range(self.config.llm_max_retries + 1):
                try:
                    report = self._generate_via_llm(
                        comm_id, entities_subset, entity_texts,
                        relations_subset, edge_texts,
                    )
                    if report.title and report.summary:
                        return self._finalize_report(
                            report, comm_id, entity_names, comm_relations,
                        )
                except Exception as e:
                    log.warning(f"[CommunitySummary] LLM generation failed "
                                f"(community {comm_id}, attempt {attempt+1}): {e}")

        # Fallback to heuristic
        if self.config.fallback_to_heuristic:
            report = self._generate_via_heuristic(
                comm_id, entity_names, entity_texts, comm_relations,
            )
            return self._finalize_report(report, comm_id, entity_names, comm_relations)

        # Return minimal report
        report = CommunityReport(
            id=str(comm_id),
            title=f"Community {comm_id}",
            summary=f"Community containing {len(entity_names)} entities.",
            entity_count=len(entity_names),
            edge_count=len(comm_relations),
        )
        return self._finalize_report(report, comm_id, entity_names, comm_relations)

    def _generate_via_llm(
        self,
        comm_id: int,
        entity_names: List[str],
        entity_texts: Dict[str, str],
        relations: List[Dict[str, Any]],
        edge_texts: Optional[Dict[str, str]],
    ) -> CommunityReport:
        """Generate community report using LLM."""
        # Build entity list text
        entity_lines = []
        for name in entity_names:
            desc = entity_texts.get(name, "")
            if desc:
                entity_lines.append(f"- {name}: {desc[:200]}")
            else:
                entity_lines.append(f"- {name}")
        entity_text = "\n".join(entity_lines)

        # Build relation list text
        relation_lines = []
        for rel in relations:
            src = rel.get("source", "?")
            tgt = rel.get("target", "?")
            desc = rel.get("description", rel.get("relation", ""))
            relation_lines.append(f"- {src} → {tgt}: {desc[:200]}")
        relation_text = "\n".join(relation_lines)

        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            community_id=comm_id,
            entity_list=entity_text,
            relation_list=relation_text,
        )

        response = self._llm.generate(prompt)

        # Parse JSON response
        title, summary, findings = self._parse_llm_response(response)

        return CommunityReport(
            id=str(comm_id),
            title=title,
            summary=summary,
            findings=findings,
        )

    def _generate_via_heuristic(
        self,
        comm_id: int,
        entity_names: List[str],
        entity_texts: Dict[str, str],
        relations: List[Dict[str, Any]],
    ) -> CommunityReport:
        """Generate community report using heuristic (no LLM).

        Creates a descriptive title from top entity names,
        statistical summary, and basic findings.
        """
        # Title: top 3 entity names concatenated
        top_entities = entity_names[:3]
        title = f"Community of {', '.join(top_entities)}"
        if len(entity_names) > 3:
            title += f" and {len(entity_names) - 3} others"

        # Summary: statistical description
        summary_parts = [
            f"This community contains {len(entity_names)} entities "
            f"and {len(relations)} relationships.",
        ]
        # Add top entity descriptions
        for name in top_entities:
            desc = entity_texts.get(name, "")
            if desc:
                summary_parts.append(f"{name}: {desc[:100]}.")

        summary = " ".join(summary_parts)

        # Findings: basic structural observations
        findings = []

        # Finding 1: Size
        findings.append(CommunityFinding(
            summary=f"Community size: {len(entity_names)} entities",
            explanation=f"The community has {len(entity_names)} connected entities "
                        f"forming a cohesive subgraph.",
        ))

        # Finding 2: Hub entities (most connected)
        entity_degree = defaultdict(int)
        for rel in relations:
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            if src in entity_names:
                entity_degree[src] += 1
            if tgt in entity_names:
                entity_degree[tgt] += 1

        if entity_degree:
            hub = max(entity_degree, key=entity_degree.get)
            findings.append(CommunityFinding(
                summary=f"Hub entity: {hub} (degree {entity_degree[hub]})",
                explanation=f"{hub} is the most connected entity in this community, "
                            f"with {entity_degree[hub]} relationships.",
            ))

        # Finding 3: Relation types
        rel_types = set()
        for rel in relations:
            desc = rel.get("description", rel.get("relation", ""))
            if desc:
                rel_types.add(desc.split()[0] if desc else "unknown")
        if rel_types:
            findings.append(CommunityFinding(
                summary=f"Relationship types: {', '.join(list(rel_types)[:5])}",
                explanation=f"The community contains {len(rel_types)} distinct "
                            f"relationship types.",
            ))

        return CommunityReport(
            id=str(comm_id),
            title=title,
            summary=summary,
            findings=findings[:self.config.max_findings],
        )

    def _finalize_report(
        self,
        report: CommunityReport,
        comm_id: int,
        all_entity_names: List[str],
        all_relations: List[Dict],
    ) -> CommunityReport:
        """Finalize report: add metadata and optionally generate embedding."""
        report.id = str(comm_id)
        report.entity_count = len(all_entity_names)
        report.edge_count = len(all_relations)

        # Generate embedding for retrieval
        if self.config.generate_embeddings and self._embedding_fn:
            try:
                content = report.full_content()
                emb = self._embedding_fn(content)
                report.embedding = emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
            except Exception as e:
                log.warning(f"[CommunitySummary] Embedding failed for community {comm_id}: {e}")

        return report

    @staticmethod
    def _filter_community_relations(
        entity_names: List[str],
        all_relations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter relations that involve entities in this community."""
        entity_set = set(entity_names)
        comm_relations = []
        for rel in all_relations:
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            if src in entity_set or tgt in entity_set:
                comm_relations.append(rel)
        return comm_relations

    @staticmethod
    def _parse_llm_response(response: str) -> Tuple[str, str, List[CommunityFinding]]:
        """Parse LLM JSON response for community report."""
        # Strip markdown fences
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            payload = json.loads(text)
            title = payload.get("title", "")
            summary = payload.get("summary", "")
            findings_raw = payload.get("findings", [])
            findings = []
            for f in findings_raw:
                findings.append(CommunityFinding(
                    summary=f.get("summary", ""),
                    explanation=f.get("explanation", ""),
                ))
            return title, summary, findings
        except json.JSONDecodeError:
            # Regex fallback
            title_match = re.search(r'"title"\s*:\s*"([^"]*)"', text)
            summary_match = re.search(r'"summary"\s*:\s*"([^"]*)"', text)
            title = title_match.group(1) if title_match else ""
            summary = summary_match.group(1) if summary_match else ""
            return title, summary, []


import re  # noqa: E402 — re is used above but imported at module level already

# ── Convenience function ──────────────────────────────────────────


def generate_community_summaries(
    communities: Dict[int, List[str]],
    entity_texts: Dict[str, str],
    relations: Optional[List[Dict[str, Any]]] = None,
    llm: Optional[Any] = None,
    embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
) -> List[CommunityReport]:
    """Quick-generate community summaries from detected communities.

    Args:
        communities: Dict mapping community ID to entity names.
        entity_texts: Entity name → description mapping.
        relations: Optional list of relations.
        llm: Optional LLM for rich summaries.
        embedding_fn: Optional embedding function for VDB search.

    Returns:
        List of CommunityReport objects.
    """
    generator = CommunitySummaryGenerator(llm=llm, embedding_fn=embedding_fn)
    return generator.generate(communities, entity_texts, relations)
