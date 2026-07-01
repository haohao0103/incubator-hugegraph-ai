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

"""LightRAG-style iterative Map-Reduce description merging for entity/relation descriptions.

Adapted from LightRAG's ``_handle_entity_relation_summary()`` (operate.py) and
``summarize_entity_descriptions`` prompt template.

This operator merges conflicting or overlapping descriptions of the same entity
or relation into a single coherent summary using a 4-level strategy:

- Level 1: single description -> return directly (no LLM)
- Level 2: few short descriptions -> join with separator (no LLM)
- Level 3: moderate descriptions -> single LLM summarisation
- Level 4: many/long descriptions -> iterative Map-Reduce

It integrates with HugeGraph-AI's operator pipeline via the ``run(context)``
protocol, reading entities from context and writing merged descriptions back.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional

log = getLogger(__name__)

# ---------------------------------------------------------------------------
# CJK detection regex
# ---------------------------------------------------------------------------
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]")

# ---------------------------------------------------------------------------
# Prompt template — adapted from LightRAG's summarize_entity_descriptions
# ---------------------------------------------------------------------------
SUMMARIZE_PROMPT_TEMPLATE = """\
---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given {kind} into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided below. Each description is labelled with its source.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
4. Context: Write the summary from an objective, third-person perspective. Explicitly mention the name of the {kind} for full clarity.
5. Conflict Handling:
  - If conflicting descriptions appear to describe multiple distinct {kind}s that share the same name, summarize each one *separately* within the overall output.
  - If conflicts within a single {kind} exist (e.g., historical discrepancies), attempt to reconcile them or present both viewpoints with noted uncertainty.
6. Length Constraint: The summary's total length must not exceed {max_output_tokens} tokens, while still maintaining depth and completeness.
7. Language: The entire output must be written in {language}. Proper nouns may be retained in their original language if proper translation is not available.

---Input---
{kind} Name: {name}

Description List:

{description_list_text}

---Output---
"""

# Default language when not specified
DEFAULT_LANGUAGE = "English"

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Estimate token count for *text*.

    Uses tiktoken if available; otherwise falls back to a heuristic:
    ~3 chars/token for Latin text, ~1.5 chars/token for CJK characters.
    """
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        return len(enc.encode(text))
    except Exception:
        cjk_chars = len(CJK_PATTERN.findall(text))
        non_cjk_chars = len(text) - cjk_chars
        return int(non_cjk_chars / 3.0 + cjk_chars / 1.5 + 0.5)


def _has_cjk(text: str) -> bool:
    """Return True if *text* contains any CJK characters."""
    return bool(CJK_PATTERN.search(text))


# ---------------------------------------------------------------------------
# Description list formatting (JSONL — matches LightRAG format)
# ---------------------------------------------------------------------------

def _format_description_list(descriptions: List[str]) -> str:
    """Format description list as JSONL (one JSON object per line),
    matching LightRAG's ``summarize_entity_descriptions`` input format.
    """
    lines = []
    for i, desc in enumerate(descriptions, 1):
        lines.append(json.dumps({"Source": f"D{i}", "Description": desc}, ensure_ascii=False))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Partition helper
# ---------------------------------------------------------------------------

def _partition_descriptions(
    descriptions: List[str],
    max_tokens_per_partition: int,
) -> List[List[str]]:
    """Split *descriptions* into partitions where each partition's total
    token count stays within *max_tokens_per_partition*.

    Each description is kept intact — never split mid-text.  If a single
    description exceeds the budget, it gets its own partition.
    """
    partitions: List[List[str]] = []
    current: List[str] = []
    current_tokens = 0

    for desc in descriptions:
        desc_tokens = _count_tokens(desc)
        if not current and desc_tokens > max_tokens_per_partition:
            partitions.append([desc])
            continue
        if current and (current_tokens + desc_tokens > max_tokens_per_partition):
            partitions.append(current)
            current = []
            current_tokens = 0
        current.append(desc)
        current_tokens += desc_tokens

    if current:
        partitions.append(current)

    return partitions


# ---------------------------------------------------------------------------
# LLM summarization (Reduce phase)
# ---------------------------------------------------------------------------

def _summarize_partition(
    partition: List[str],
    llm_func: Callable[[str], str],
    kind: str = "entity",
    name: str = "unknown",
    max_output_tokens: int = 600,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """Summarize a single partition of descriptions via one LLM call."""
    description_list_text = _format_description_list(partition)

    prompt = SUMMARIZE_PROMPT_TEMPLATE.format(
        kind=kind,
        name=name,
        description_list_text=description_list_text,
        max_output_tokens=max_output_tokens,
        language=language,
    )
    return llm_func(prompt).strip()


async def _summarize_partition_async(
    partition: List[str],
    llm_func_async: Callable,
    kind: str = "entity",
    name: str = "unknown",
    max_output_tokens: int = 600,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """Async version of :func:`_summarize_partition`."""
    description_list_text = _format_description_list(partition)

    prompt = SUMMARIZE_PROMPT_TEMPLATE.format(
        kind=kind,
        name=name,
        description_list_text=description_list_text,
        max_output_tokens=max_output_tokens,
        language=language,
    )
    if asyncio.iscoroutinefunction(llm_func_async):
        result = await llm_func_async(prompt)
    else:
        result = llm_func_async(prompt)
    return result.strip()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DescriptionMergerConfig:
    """Configuration for :class:`DescriptionMerger`.

    Defaults match LightRAG's original thresholds.
    """

    force_llm_threshold: int = 8
    summary_max_tokens: int = 1200
    summary_context_size: int = 12000
    max_output_tokens: int = 600
    separator: str = "\n"
    kind: str = "entity"
    name: str = "unknown"
    language: str = DEFAULT_LANGUAGE


# ---------------------------------------------------------------------------
# Main merger class
# ---------------------------------------------------------------------------

class DescriptionMerger:
    """Iterative Map-Reduce description merger (LightRAG-style).

    Implements a 4-level strategy:

    * **Level 1** -- single description -> return directly (no LLM).
    * **Level 2** -- few descriptions whose total tokens < *summary_max_tokens*
      AND count < *force_llm_threshold* -> join with separator (no LLM).
    * **Level 3** -- descriptions whose total tokens <= *summary_context_size*
      -> single LLM summarisation.
    * **Level 4** -- too many tokens -> iterative Map-Reduce until Level 2 or 3
      is satisfied.

    Parameters
    ----------
    config : DescriptionMergerConfig, optional
        Merging thresholds and output limits.
    llm_func : callable or None
        ``llm_func(prompt: str) -> str``.  If *None*, only Level 1 and 2 are
        available; Levels 3/4 fall back to simple join.
    """

    def __init__(
        self,
        config: Optional[DescriptionMergerConfig] = None,
        llm_func: Optional[Callable[[str], str]] = None,
    ):
        self.config = config or DescriptionMergerConfig()
        self.llm_func = llm_func

    # -- public API ----------------------------------------------------------

    def merge(self, descriptions: List[str]) -> str:
        """Merge *descriptions* synchronously."""
        if not descriptions:
            return ""
        descriptions = [d for d in descriptions if d.strip()]
        if not descriptions:
            return ""
        return self._merge_levels(descriptions)

    async def merge_async(
        self,
        descriptions: List[str],
        llm_func_async: Optional[Callable] = None,
    ) -> str:
        """Merge *descriptions* asynchronously."""
        if not descriptions:
            return ""
        descriptions = [d for d in descriptions if d.strip()]
        if not descriptions:
            return ""
        async_func = llm_func_async or self.llm_func
        return await self._merge_levels_async(descriptions, async_func)

    # -- HG-AI operator protocol -------------------------------------------

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Operator protocol: merge descriptions for entities in context.

        Reads from context:
            vertices: List of vertex dicts with "properties" containing
                      "description" fields.  Can come from PropertyGraphExtract
                      or EntityResolution output.
            resolution_result: Optional. If present, contains ``merged_pairs``
                      from EntityResolution -- descriptions of merged-away entities
                      should be merged into the surviving entity.

        Writes to context:
            merged_descriptions: Dict mapping entity_name -> merged_description.
            vertices: Updated vertex list with merged descriptions (in-place).
        """
        vertices = context.get("vertices", [])
        resolution_result = context.get("resolution_result", {})

        # Collect all descriptions per entity name
        entity_descriptions: Dict[str, List[str]] = {}
        for vertex in vertices:
            props = vertex.get("properties", {})
            name = props.get("name", "")
            desc = props.get("description", "")
            if name and desc:
                entity_descriptions.setdefault(name, []).append(desc)

        # If EntityResolution produced merged_pairs, add descriptions from
        # deprecated entities into the surviving entity's list.
        merged_pairs = resolution_result.get("merged_pairs", [])
        for pair in merged_pairs:
            from_name = pair.get("from_name", "")
            to_name = pair.get("to_name", "")
            from_desc = pair.get("from_description", "")
            to_desc = pair.get("to_description", "")
            if to_name and to_desc:
                entity_descriptions.setdefault(to_name, []).append(to_desc)
            if to_name and from_desc:
                entity_descriptions.setdefault(to_name, []).append(from_desc)

        # Merge descriptions for each entity
        merged: Dict[str, str] = {}
        for entity_name, descs in entity_descriptions.items():
            if len(descs) <= 1:
                merged[entity_name] = descs[0] if descs else ""
            else:
                entity_config = DescriptionMergerConfig(
                    force_llm_threshold=self.config.force_llm_threshold,
                    summary_max_tokens=self.config.summary_max_tokens,
                    summary_context_size=self.config.summary_context_size,
                    max_output_tokens=self.config.max_output_tokens,
                    separator=self.config.separator,
                    kind="entity",
                    name=entity_name,
                    language=self.config.language,
                )
                merger = DescriptionMerger(config=entity_config, llm_func=self.llm_func)
                merged[entity_name] = merger.merge(descs)

        # Update vertices in-place
        for vertex in vertices:
            props = vertex.get("properties", {})
            name = props.get("name", "")
            if name in merged and merged[name] != props.get("description", ""):
                props["description"] = merged[name]

        context["merged_descriptions"] = merged
        log.info("DescriptionMerger: merged descriptions for %d entities", len(merged))
        return context

    # -- internal: level dispatch -------------------------------------------

    def _classify_level(self, descriptions: List[str]) -> int:
        """Determine which merge level applies."""
        count = len(descriptions)
        if count == 1:
            return 1

        total_tokens = sum(_count_tokens(d) for d in descriptions)

        if count < self.config.force_llm_threshold and total_tokens < self.config.summary_max_tokens:
            return 2

        if total_tokens <= self.config.summary_context_size:
            return 3

        return 4

    def _merge_levels(self, descriptions: List[str]) -> str:
        level = self._classify_level(descriptions)

        if level == 1:
            return descriptions[0]

        if level == 2:
            return self.config.separator.join(descriptions)

        if level == 3:
            if self.llm_func is None:
                return self.config.separator.join(descriptions)
            return _summarize_partition(
                descriptions,
                self.llm_func,
                kind=self.config.kind,
                name=self.config.name,
                max_output_tokens=self.config.max_output_tokens,
                language=self.config.language,
            )

        # Level 4 -- iterative Map-Reduce
        return self._iterative_map_reduce(descriptions)

    async def _merge_levels_async(
        self, descriptions: List[str], llm_func_async: Optional[Callable]
    ) -> str:
        level = self._classify_level(descriptions)

        if level == 1:
            return descriptions[0]

        if level == 2:
            return self.config.separator.join(descriptions)

        if level == 3:
            if llm_func_async is None:
                return self.config.separator.join(descriptions)
            return await _summarize_partition_async(
                descriptions,
                llm_func_async,
                kind=self.config.kind,
                name=self.config.name,
                max_output_tokens=self.config.max_output_tokens,
                language=self.config.language,
            )

        # Level 4 -- iterative async Map-Reduce
        return await self._iterative_map_reduce_async(descriptions, llm_func_async)

    # -- iterative Map-Reduce -----------------------------------------------

    def _iterative_map_reduce(self, descriptions: List[str]) -> str:
        """Synchronous iterative Map-Reduce (Level 4)."""
        if self.llm_func is None:
            return self.config.separator.join(descriptions)

        current = descriptions
        while True:
            level = self._classify_level(current)
            if level <= 3:
                return self._merge_levels(current)

            partitions = _partition_descriptions(
                current, self.config.summary_context_size
            )
            summaries: List[str] = []
            for partition in partitions:
                summary = _summarize_partition(
                    partition,
                    self.llm_func,
                    kind=self.config.kind,
                    name=self.config.name,
                    max_output_tokens=self.config.max_output_tokens,
                    language=self.config.language,
                )
                summaries.append(summary)
            current = summaries

    async def _iterative_map_reduce_async(
        self, descriptions: List[str], llm_func_async: Optional[Callable]
    ) -> str:
        """Async iterative Map-Reduce (Level 4) with cooperative yield."""
        if llm_func_async is None:
            return self.config.separator.join(descriptions)

        current = descriptions
        processed = 0
        while True:
            level = self._classify_level(current)
            if level <= 3:
                return await self._merge_levels_async(current, llm_func_async)

            partitions = _partition_descriptions(
                current, self.config.summary_context_size
            )
            summaries: List[str] = []
            for partition in partitions:
                processed += len(partition)
                summary = await _summarize_partition_async(
                    partition,
                    llm_func_async,
                    kind=self.config.kind,
                    name=self.config.name,
                    max_output_tokens=self.config.max_output_tokens,
                    language=self.config.language,
                )
                summaries.append(summary)
                if processed % 32 == 0:
                    await asyncio.sleep(0)
            current = summaries
