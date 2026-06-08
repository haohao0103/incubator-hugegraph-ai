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

"""LLM-based community report generation.

Generates structured natural language summaries for each detected community,
following the Microsoft GraphRAG community report pattern.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hugegraph_llm.config import prompt
from hugegraph_llm.utils.log import log


@dataclass
class CommunityReport:
    """A structured report for a single community."""

    community_id: str
    level: int
    title: str = ""
    summary: str = ""
    key_entities: List[str] = field(default_factory=list)
    relationship_patterns: List[str] = field(default_factory=list)
    importance_score: float = 0.0
    source_vertices: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "community_id": self.community_id,
            "level": self.level,
            "title": self.title,
            "summary": self.summary,
            "key_entities": self.key_entities,
            "relationship_patterns": self.relationship_patterns,
            "importance_score": self.importance_score,
            "source_vertices": self.source_vertices,
        }

    def to_text(self) -> str:
        """Format the report as a compact text string for LLM context."""
        return (
            f"[{self.title}] (importance: {self.importance_score:.1f})\n"
            f"Key Entities: {', '.join(self.key_entities[:5])}\n"
            f"Patterns: {'; '.join(self.relationship_patterns[:3])}\n"
            f"Summary: {self.summary}"
        )


class CommunityReportGenerate:
    """Generate LLM-based structured reports for graph communities.

    For each community detected by CommunityDetect, generates a
    structured report including:
    - Community title (concise name)
    - Key entities (top 3-5 most important)
    - Relationship patterns (key connection structures)
    - Summary (2-3 sentence description)
    - Importance score (0.0-10.0 based on density and relevance)

    Reports are generated in parallel batches to handle large numbers
    of communities efficiently.

    Usage:
        reporter = CommunityReportGenerate(llm=chat_llm)
        context = reporter.run(context)
        # context["community_reports"] = [CommunityReport(...), ...]
    """

    # Prompt template for community report generation
    REPORT_PROMPT_TEMPLATE = """You are analyzing a community of connected entities from a knowledge graph.

## Community Information
Community ID: {community_id}
Level: {level}
Size: {size} entities, {edge_count} internal connections
Density: {density:.4f}

### Entities in this community:
{entity_list}

### Relationships in this community:
{relationship_list}

## Task
Generate a structured analysis of this community. Output ONLY valid JSON:

{{
  "title": "A concise, descriptive name for this community (3-8 words)",
  "summary": "A 2-3 sentence description of what this community represents",
  "key_entities": ["most important entity 1", "entity 2", "entity 3"],
  "relationship_patterns": ["key pattern 1", "key pattern 2"],
  "importance_score": <0.0 to 10.0 based on density and information value>
}}

## Guidelines
- Title should capture the theme or topic of this group
- Key entities should be the most central/connected nodes
- Relationship patterns describe recurring connection structures
- Importance: dense communities with rich relationships score higher
"""

    MAX_ENTITY_LIST = 50
    MAX_RELATIONSHIP_LIST = 30
    BATCH_SIZE = 5  # Concurrent LLM calls per batch
    MAX_CONCURRENCY = 10  # Hard cap on concurrent LLM API calls

    def __init__(
        self,
        llm: Any = None,
        max_communities: int = 100,
    ):
        """Initialize the community report generator.

        Args:
            llm: LLM instance for generating reports (extract_llm recommended).
            max_communities: Maximum number of communities to generate reports for.
        """
        self._llm = llm
        self._max_communities = max_communities

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate community reports from detected communities.

        Reads from context:
            communities: List of community dicts (from CommunityDetect).
            nx_graph: The networkx graph (for density context, optional).

        Writes to context:
            community_reports: List of CommunityReport objects.
        """
        communities = context.get("communities", [])
        if not communities:
            log.warning("No communities found. Skipping report generation.")
            context["community_reports"] = []
            return context

        # Limit number of communities
        communities = communities[: self._max_communities]

        # Sort by size (larger communities first)
        communities = sorted(communities, key=lambda c: c.get("size", 0), reverse=True)

        # Generate reports in batches
        reports = []
        for i in range(0, len(communities), self.BATCH_SIZE):
            batch = communities[i : i + self.BATCH_SIZE]
            batch_reports = self._generate_batch(batch)
            reports.extend(batch_reports)
            log.debug(
                "Generated reports for batch %d-%d/%d",
                i + 1,
                min(i + self.BATCH_SIZE, len(communities)),
                len(communities),
            )

        # Sort by importance score
        reports.sort(key=lambda r: r.importance_score, reverse=True)

        context["community_reports"] = [r.to_dict() for r in reports]
        log.info("Generated %d community reports", len(reports))
        return context

    def _generate_batch(self, batch: List[Dict]) -> List[CommunityReport]:
        """Generate reports for a batch of communities in parallel.

        Uses asyncio.Semaphore to cap concurrent LLM API calls at
        MAX_CONCURRENCY, preventing API rate-limit errors while still
        parallelizing within the batch.
        """
        if not self._llm:
            # No LLM available → all fallback, no need for async
            return [self._fallback_report(c) for c in batch]

        semaphore = asyncio.Semaphore(self.MAX_CONCURRENCY)

        async def _safe_generate(comm: Dict) -> CommunityReport:
            async with semaphore:
                try:
                    # LLM generate is blocking, run in thread
                    return await asyncio.get_event_loop().run_in_executor(
                        None, self._generate_single, comm
                    )
                except Exception as e:
                    log.error(
                        "Failed to generate report for community %s: %s",
                        comm.get("id", "unknown"),
                        str(e),
                    )
                    return self._fallback_report(comm)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        tasks = [_safe_generate(comm) for comm in batch]
        return loop.run_until_complete(asyncio.gather(*tasks))

    def _generate_single(self, comm: Dict) -> CommunityReport:
        """Generate a report for a single community via LLM."""
        # Format entity list
        vertex_details = comm.get("vertex_details", [])
        entity_lines = []
        for v in vertex_details[: self.MAX_ENTITY_LIST]:
            label = v.get("label", "unknown")
            props = v.get("props", {})
            name = props.get("name", v.get("id", "?"))
            entity_lines.append(f"  - [{label}] {name}")
        entity_list = "\n".join(entity_lines) if entity_lines else "(no entities)"

        # Format relationship list
        edge_details = comm.get("edge_details", [])
        rel_lines = []
        for e in edge_details[: self.MAX_RELATIONSHIP_LIST]:
            label = e.get("label", "")
            out_v = e.get("outV", "")
            in_v = e.get("inV", "")
            rel_lines.append(f"  - ({out_v})-[{label}]->({in_v})")
        relationship_list = "\n".join(rel_lines) if rel_lines else "(no relationships)"

        # Build prompt
        report_prompt = self.REPORT_PROMPT_TEMPLATE.format(
            community_id=comm.get("id", "?"),
            level=comm.get("level", 0),
            size=comm.get("size", 0),
            edge_count=len(edge_details),
            density=comm.get("density", 0.0),
            entity_list=entity_list,
            relationship_list=relationship_list,
        )

        if self._llm is None:
            return self._fallback_report(comm)

        response = self._llm.generate(prompt=report_prompt)
        parsed = self._parse_response(response)

        return CommunityReport(
            community_id=comm.get("id", "unknown"),
            level=comm.get("level", 0),
            title=parsed.get("title", f"Community {comm.get('id', '?')}"),
            summary=parsed.get("summary", ""),
            key_entities=parsed.get("key_entities", []),
            relationship_patterns=parsed.get("relationship_patterns", []),
            importance_score=float(parsed.get("importance_score", 5.0)),
            source_vertices=comm.get("vertices", []),
        )

    @staticmethod
    def _parse_response(response: str) -> Dict[str, Any]:
        """Parse LLM response JSON, handling markdown code fences and errors."""
        # Try to extract JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try raw JSON
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            json_str = json_match.group(0) if json_match else response

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            log.warning("Failed to parse community report JSON: %s", response[:200])
            return {
                "title": "Unknown Community",
                "summary": response[:300],
                "key_entities": [],
                "relationship_patterns": [],
                "importance_score": 3.0,
            }

    @staticmethod
    def _fallback_report(comm: Dict) -> CommunityReport:
        """Generate a basic report without LLM (for error cases)."""
        vertex_details = comm.get("vertex_details", [])
        labels = set(v.get("label", "unknown") for v in vertex_details)
        top_entities = [
            v.get("props", {}).get("name", v.get("id", "?"))
            for v in vertex_details[:3]
        ]

        return CommunityReport(
            community_id=comm.get("id", "unknown"),
            level=comm.get("level", 0),
            title=f"Community: {', '.join(sorted(labels)[:3])}",
            summary=(
                f"A community of {comm.get('size', 0)} connected entities "
                f"primarily of type(s): {', '.join(sorted(labels))}."
            ),
            key_entities=top_entities,
            relationship_patterns=[],
            importance_score=3.0,
            source_vertices=comm.get("vertices", []),
        )
