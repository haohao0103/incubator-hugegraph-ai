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

"""MapReduce Global Search over community reports.

Implements Microsoft GraphRAG's Global Search pattern:
1. MAP: Match query to relevant communities, generate per-community point-form answers
2. REDUCE: Synthesize all intermediate answers into a final comprehensive response
"""

import json
import random
from typing import Any, Dict, List, Optional

from hugegraph_llm.utils.log import log


class GlobalSearch:
    """MapReduce-style global search over community reports.

    This enables answering broad, thematic questions about the entire
    knowledge graph by operating on pre-computed community summaries
    rather than individual entities/relationships.

    Two-phase approach:
    1. MAP: For each relevant community, the LLM generates a focused
       point-form answer with an importance score (0-10).
    2. REDUCE: All intermediate answers are ranked, filtered, and
       synthesized into a single comprehensive response.

    Usage:
        searcher = GlobalSearch(llm=chat_llm)
        context = searcher.run(context)
        # context["global_answer"] = "Comprehensive answer..."
    """

    MAP_PROMPT_TEMPLATE = """You are analyzing community reports from a knowledge graph to answer a user's question.

## User's Question
{query}

## Community Report ({community_title}, importance: {importance_score:.1f})
{report_text}

## Task
Based on this community report, generate 3-5 point-form key findings that help answer the user's question.
Each finding MUST be tagged with an importance score (0.0-10.0).

Output format:
Finding: <key finding>
Score: <0.0-10.0>

Finding: <key finding>
Score: <0.0-10.0>

Do NOT include: any findings that are not relevant to the question, filler text, or explanations.
"""

    REDUCE_PROMPT_TEMPLATE = """You are synthesizing a comprehensive answer from multiple community-level analyses.

## User's Question
{query}

## Community-Level Findings (sorted by importance)
{findings_text}

## Task
Synthesize these findings into ONE comprehensive, well-structured answer.
- Integrate findings from different communities
- Identify overarching themes and patterns
- Provide a coherent narrative that answers the user's question
- Include specific examples from the findings
- Write in a professional, analytical tone

Answer:
"""

    MAX_MAP_COMMUNITIES = 20  # Max communities to run MAP on
    MAX_MAP_FINDINGS = 5  # Max findings per community
    MIN_SCORE_THRESHOLD = 1.0  # Filter out findings below this score
    BATCH_SIZE = 5  # Parallel processing batch size

    def __init__(
        self,
        llm: Any = None,
        max_map_communities: int = 20,
        min_score_threshold: float = 1.0,
    ):
        """Initialize the global search engine.

        Args:
            llm: LLM instance for MAP and REDUCE phases (chat_llm recommended).
            max_map_communities: Max communities to include in MAP phase.
            min_score_threshold: Minimum score to include finding in REDUCE phase.
        """
        self._llm = llm
        self._max_map_communities = max_map_communities
        self._min_score_threshold = min_score_threshold

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Global Search over community reports.

        Reads from context:
            query: The user's question.
            community_reports: List of community report dicts.
            community_matches: Optional pre-filtered community IDs.

        Writes to context:
            global_answer: The final synthesized answer.
            map_findings: Intermediate findings from MAP phase.
            communities_used: How many communities were used.
        """
        query = context.get("query", "")
        community_reports = context.get("community_reports", [])

        if not community_reports:
            log.warning("No community reports available for global search.")
            context["global_answer"] = (
                "Community reports have not been generated yet. "
                "Please run community detection first."
            )
            context["map_findings"] = []
            context["communities_used"] = 0
            return context

        # Phase 1: MAP
        map_findings = self._map_phase(query, community_reports)

        # Phase 2: REDUCE
        if not map_findings:
            context["global_answer"] = (
                "No relevant community-level information was found "
                "to answer this question."
            )
            context["map_findings"] = []
            context["communities_used"] = 0
            return context

        global_answer = self._reduce_phase(query, map_findings)

        context["global_answer"] = global_answer
        context["map_findings"] = map_findings
        context["communities_used"] = len(
            set(f.get("community_id") for f in map_findings)
        )
        log.info(
            "Global search: %d findings from %d communities",
            len(map_findings),
            context["communities_used"],
        )
        return context

    # ── MAP Phase ─────────────────────────────────────────────

    def _map_phase(
        self, query: str, community_reports: List[Dict]
    ) -> List[Dict[str, Any]]:
        """MAP: Generate point-form findings from each relevant community.

        Communities are selected based on:
        1. Importance score (top-N most important communities)
        2. Semantic relevance to query (if community index available)

        Args:
            query: The user's question.
            community_reports: All community report dicts.

        Returns:
            Ranked list of finding dicts.
        """
        # Select top communities by importance
        selected = sorted(
            community_reports,
            key=lambda r: r.get("importance_score", 0),
            reverse=True,
        )[: self._max_map_communities]

        # Shuffle to reduce bias in batch processing
        random.shuffle(selected)

        all_findings = []
        for i in range(0, len(selected), self.BATCH_SIZE):
            batch = selected[i : i + self.BATCH_SIZE]
            batch_findings = self._map_batch(query, batch)
            all_findings.extend(batch_findings)

        # Filter by minimum score and sort
        all_findings = [
            f
            for f in all_findings
            if f.get("score", 0) >= self._min_score_threshold
        ]
        all_findings.sort(key=lambda f: f.get("score", 0), reverse=True)

        log.debug("MAP phase: %d total findings after filtering", len(all_findings))
        return all_findings

    def _map_batch(
        self, query: str, batch: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Process a batch of communities in MAP phase."""
        findings = []

        for report in batch:
            # Build the community text representation
            report_text = (
                f"Title: {report.get('title', 'Unknown')}\n"
                f"Summary: {report.get('summary', '')}\n"
                f"Key Entities: {', '.join(report.get('key_entities', []))}\n"
                f"Patterns: {'; '.join(report.get('relationship_patterns', []))}"
            )

            map_prompt = self.MAP_PROMPT_TEMPLATE.format(
                query=query,
                community_title=report.get("title", "Unknown"),
                importance_score=report.get("importance_score", 5.0),
                report_text=report_text,
            )

            try:
                if self._llm:
                    response = self._llm.generate(prompt=map_prompt)
                else:
                    response = ""
                parsed = self._parse_findings(
                    response, report.get("community_id", ""), report.get("title", "")
                )
                findings.extend(parsed)
            except Exception as e:
                log.error(
                    "MAP failed for community %s: %s",
                    report.get("community_id", ""),
                    str(e),
                )

        return findings

    @staticmethod
    def _parse_findings(
        response: str, community_id: str, community_title: str
    ) -> List[Dict[str, Any]]:
        """Parse the LLM response in 'Finding: ... Score: ...' format."""
        findings = []
        lines = response.strip().split("\n")

        current_finding = None
        for line in lines:
            line = line.strip()
            if line.lower().startswith("finding:"):
                if current_finding and current_finding.get("finding"):
                    findings.append(current_finding)
                current_finding = {
                    "community_id": community_id,
                    "community_title": community_title,
                    "finding": line[len("finding:"):].strip(),
                    "score": 5.0,
                }
            elif line.lower().startswith("score:"):
                if current_finding:
                    try:
                        current_finding["score"] = float(
                            line[len("score:"):].strip()
                        )
                    except ValueError:
                        pass

        # Don't forget the last finding
        if current_finding and current_finding.get("finding"):
            findings.append(current_finding)

        return findings

    # ── REDUCE Phase ──────────────────────────────────────────

    def _reduce_phase(
        self, query: str, map_findings: List[Dict[str, Any]]
    ) -> str:
        """REDUCE: Synthesize all MAP findings into a final answer.

        Args:
            query: The original user question.
            map_findings: Ranked list of finding dicts from MAP phase.

        Returns:
            Final synthesized answer string.
        """
        # Format findings for the REDUCE prompt
        finding_lines = []
        for i, f in enumerate(map_findings[:50], 1):  # Max 50 findings in context
            title = f.get("community_title", "Unknown")
            score = f.get("score", 0)
            finding = f.get("finding", "")
            finding_lines.append(
                f"{i}. [{title}] (score: {score:.1f})\n   {finding}"
            )

        findings_text = "\n\n".join(finding_lines)

        reduce_prompt = self.REDUCE_PROMPT_TEMPLATE.format(
            query=query,
            findings_text=findings_text,
        )

        if self._llm is None:
            return (
                "Global search requires an LLM for synthesis. "
                f"{len(map_findings)} findings were collected."
            )

        try:
            answer = self._llm.generate(prompt=reduce_prompt)
            return answer.strip()
        except Exception as e:
            log.error("REDUCE phase failed: %s", str(e))
            # Fallback: concatenate top findings
            top_findings = "\n".join(
                f"- {f['finding']}" for f in map_findings[:5]
            )
            return (
                f"Based on community analysis, the key findings are:\n\n{top_findings}"
            )
