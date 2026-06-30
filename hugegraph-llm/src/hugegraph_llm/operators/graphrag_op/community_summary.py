# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not in this file except in compliance
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
Hierarchical community summary generation for GraphRAG.

Generates LLM summaries for each community in the hierarchy,
enabling global-level query answering through Map-Reduce over
community summaries.

Inspired by Microsoft GraphRAG's Community Report approach,
adapted for HugeGraph's architecture.
"""

import json
from typing import Any, Dict, List, Optional

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.utils.log import log

# Prompt template for community summary generation
COMMUNITY_SUMMARY_PROMPT_EN = """You are an expert knowledge graph analyst. Your task is to generate a comprehensive summary of a community (cluster) of related entities in a knowledge graph.

## Community Information
- Community ID: {community_id}
- Community Size: {community_size} entities
- Community Density: {community_density:.2f}

## Community Members and Their Relationships
{community_data}

## Task
Generate a concise but comprehensive summary of this community that captures:
1. The key entities and their roles
2. The main relationships and patterns within the community
3. Any notable insights or themes that emerge from the entity relationships

## Output Format
Provide a structured summary in the following JSON format:
{{
  "title": "A descriptive title for this community",
  "summary": "A 2-4 sentence summary of the community",
  "key_entities": ["List of most important entities"],
  "key_relationships": ["List of most important relationships"],
  "themes": ["List of main themes or patterns"]
}}

Summary:"""

COMMUNITY_SUMMARY_PROMPT_CN = """你是知识图谱分析专家。你的任务是为知识图谱中的一个社区（聚类）生成综合摘要。

## 社区信息
- 社区ID: {community_id}
- 社区规模: {community_size} 个实体
- 社区密度: {community_density:.2f}

## 社区成员及其关系
{community_data}

## 任务
生成一个简洁但全面的社区摘要，涵盖：
1. 关键实体及其角色
2. 社区内的主要关系和模式
3. 从实体关系中得出的重要洞察或主题

## 输出格式
请按以下 JSON 格式提供结构化摘要：
{{
  "title": "此社区的描述性标题",
  "summary": "2-4句社区摘要",
  "key_entities": ["最重要实体的列表"],
  "key_relationships": ["最重要关系的列表"],
  "themes": ["主要主题或模式的列表"]
}}

摘要："""


class CommunitySummarizer:
    """
    Generate hierarchical community summaries for GraphRAG.

    For each community detected by CommunityDetector, generates a
    structured summary using LLM (or a template-based fallback when
    LLM is unavailable). These summaries enable global query answering
    through Map-Reduce aggregation.
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        language: str = "en",
        max_community_data_length: int = 4000,
        use_template_fallback: bool = True,
    ):
        """
        Args:
            llm: LLM instance for summary generation.
            language: Language for summaries ('en' or 'zh').
            max_community_data_length: Max chars of community data to include in prompt.
            use_template_fallback: If True, use template-based summaries when LLM unavailable.
        """
        self.llm = llm
        self.language = language
        self.max_community_data_length = max_community_data_length
        self.use_template_fallback = use_template_fallback

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summaries for all communities in the context.

        Args:
            context: Dict with 'communities' and 'community_hierarchy'
                     from CommunityDetector, and optionally graph data.

        Returns:
            Updated context with 'community_summaries'.
        """
        communities = context.get("communities", [])
        hierarchy = context.get("community_hierarchy", {})

        if not communities:
            log.warning("No communities to summarize")
            context["community_summaries"] = []
            return context

        summaries = []

        # Summarize each community
        for i, community in enumerate(communities):
            community_id = f"C{i}"
            community_data = self._format_community_data(community, context)

            if self.llm:
                summary = self._generate_llm_summary(community_id, community, community_data)
            elif self.use_template_fallback:
                summary = self._generate_template_summary(community_id, community, community_data)
            else:
                log.warning("No LLM available and template fallback disabled, skipping community %s", community_id)
                continue

            summary["community_id"] = community_id
            summary["community_size"] = len(community)
            summaries.append(summary)

        # Also summarize hierarchy levels
        hierarchy_summaries = self._summarize_hierarchy_levels(hierarchy, summaries)
        context["community_summaries"] = summaries
        context["hierarchy_summaries"] = hierarchy_summaries
        context["call_count"] = context.get("call_count", 0) + len(summaries)

        log.info(
            "Generated %d community summaries and %d hierarchy level summaries",
            len(summaries),
            len(hierarchy_summaries),
        )
        return context

    def _format_community_data(self, community: List[str], context: Dict[str, Any]) -> str:
        """
        Format community data for inclusion in the summary prompt.

        Produces a readable representation of the community's entities
        and their relationships.
        """
        lines = []
        community_set = set(community)

        # List entities in the community
        lines.append("Entities:")
        for entity_id in community:
            label = ""
            # Try to find entity info from context
            for vertex in context.get("vertices", []):
                if str(vertex.get("id", "")) == entity_id or vertex.get("name", "") == entity_id:
                    label = vertex.get("label", "")
                    props = vertex.get("properties", {})
                    props_str = ", ".join(f"{k}: {v}" for k, v in props.items() if v)
                    lines.append(f"  - {entity_id} ({label}): {props_str}")
                    break
            else:
                lines.append(f"  - {entity_id}")

        # List relationships within the community
        lines.append("\nRelationships:")
        for edge in context.get("edges", []):
            source = str(edge.get("outV", edge.get("start", "")))
            target = str(edge.get("inV", edge.get("end", "")))
            if source in community_set and target in community_set:
                label = edge.get("label", edge.get("type", "related_to"))
                lines.append(f"  - {source} --[{label}]--> {target}")

        # Also check graph_result for relationships
        for item in context.get("graph_result", []):
            if isinstance(item, str) and any(e in item for e in community):
                lines.append(f"  - {item}")

        data = "\n".join(lines)
        if len(data) > self.max_community_data_length:
            data = data[: self.max_community_data_length] + "\n... (truncated)"
        return data

    def _generate_llm_summary(self, community_id: str, community: List[str], community_data: str) -> Dict[str, Any]:
        """Generate a community summary using LLM."""
        prompt_template = COMMUNITY_SUMMARY_PROMPT_CN if self.language == "zh" else COMMUNITY_SUMMARY_PROMPT_EN

        prompt = prompt_template.format(
            community_id=community_id,
            community_size=len(community),
            community_density=0.0,  # Will be overridden if we have density info
            community_data=community_data,
        )

        try:
            response = self.llm.generate(prompt=prompt)
            return self._parse_summary_response(response)
        except Exception as e:  # pylint: disable=broad-except
            log.error("LLM summary generation failed for community %s: %s", community_id, e)
            if self.use_template_fallback:
                return self._generate_template_summary(community_id, community, community_data)
            return {"title": f"Community {community_id}", "summary": "", "error": str(e)}

    def _generate_template_summary(
        self, community_id: str, community: List[str], community_data: str
    ) -> Dict[str, Any]:
        """Generate a basic template-based summary when LLM is unavailable."""
        top_entities = community[:5]
        title = f"Community {community_id} ({len(community)} entities)"
        summary = f"A community of {len(community)} entities including {', '.join(top_entities[:3])}"
        if len(community) > 3:
            summary += f" and {len(community) - 3} others."

        return {
            "title": title,
            "summary": summary,
            "key_entities": top_entities,
            "key_relationships": [],
            "themes": [],
        }

    def _parse_summary_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM summary response into structured dict."""
        try:
            json_match = __import__("re").search(r"\{.*\}", response, __import__("re").DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            log.warning("Failed to parse LLM summary as JSON, using raw response")

        return {
            "title": "Unparsed Summary",
            "summary": response[:500],
            "key_entities": [],
            "key_relationships": [],
            "themes": [],
        }

    def _summarize_hierarchy_levels(
        self, hierarchy: Dict[str, Any], community_summaries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate summaries for each level of the community hierarchy.

        Higher levels represent broader themes across multiple communities.
        """
        hierarchy_summaries = []
        levels = hierarchy.get("levels", [])

        for level_info in levels:
            level_num = level_info["level"]
            level_communities = level_info.get("communities", [])

            level_summary = {
                "level": level_num,
                "community_count": level_info.get("community_count", 0),
                "total_entities": sum(c.get("size", 0) for c in level_communities),
                "avg_density": 0.0,
                "themes": [],
            }

            # Compute average density
            densities = [c.get("density", 0.0) for c in level_communities]
            if densities:
                level_summary["avg_density"] = sum(densities) / len(densities)

            # Aggregate themes from community summaries
            all_themes = []
            for summary in community_summaries:
                all_themes.extend(summary.get("themes", []))
            level_summary["themes"] = list(set(all_themes))[:10]

            hierarchy_summaries.append(level_summary)

        return hierarchy_summaries
