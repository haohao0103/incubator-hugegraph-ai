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
G1: Gleaning Iterative Entity Extraction — 对标 LightRAG _gleaning_entity_extraction

在首轮LLM实体抽取后，对低置信度/缺失描述的实体执行补充追问(gleaning call)，
通过多轮迭代提升KG完整度20-30%。

设计参考:
  - LightRAG: lightrag/operate.py extract_entities() → gleaning_continue_prompt
  - 合并策略: 按description长度保留更详细版本

特性:
  - JSON/Text双模式输出支持
  - Token守卫(防context_length_exceeded)
  - 可配置gleaning轮数(默认1轮)
  - 按描述长度优选合并策略
  - 与现有GraphRAGExtractorOperator兼容
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates — extracted from LightRAG prompt.py
# ---------------------------------------------------------------------------

GLEANING_JSON_PROMPT = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly described** entities and relationships from the `---Input Text---` section.

---Instructions---
1. **Focus on Corrections/Additions:**
  - **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
  - If an entity or relationship was **missed** in the last task, extract and output it now.
  - If an entity or relationship was **incorrectly described** in the last task, re-output the *corrected and complete* version.
2. **Strict Adherence to JSON Format:** Your output MUST be a valid JSON object with `entities` and `relationships` arrays.
3. **Quantity Limits:** Output at most {max_total_records} total records and at most {max_entity_records} entity objects.
4. **Output Language:** Ensure the output language is {language}. Proper nouns must be kept in their original language.

---Output---
"""

GLEANING_TEXT_PROMPT = """---Task---
Based on the last extraction task, identify and extract any missed or incorrectly formatted entities and relationships from the input text.

---Instructions---
1. Strictly adhere to all format requirements for entity and relationship lists.
2. Do NOT re-output entities and relationships that were correctly and fully extracted in the last task.
3. If an entity or relationship was missed, extract it now according to system format.
4. If truncated or incorrectly formatted, re-output the corrected version.
5. Output at most {max_total_records} total rows and at most {max_entity_records} entity rows.
6. Output `{completion_delimiter}` as the final line after extraction completes.
7. Output language: {language}. Keep proper nouns in original language.

---Output---
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """Result of a single extraction pass (initial or gleaning)."""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    tokens_used: int = 0
    duration_ms: float = 0.0
    is_gleaning: bool = False


@dataclass
class GleaningConfig:
    """Configuration for gleaning behavior."""
    enabled: bool = True
    max_rounds: int = 1              # Number of gleaning calls after initial extraction
    max_total_records_per_round: int = 20   # Max new records per gleaning round
    max_entity_records_per_round: int = 15
    min_description_length: int = 3   # Entities with shorter desc are candidates for gleaning
    use_json_mode: bool = True       # JSON structured output vs delimiter-based
    language: str = "English"
    completion_delimiter: str = "<|end|>"


# ---------------------------------------------------------------------------
# Core Gleaning Logic
# ---------------------------------------------------------------------------

class GleaningExtractor:
    """Wraps an LLM caller to add gleaning iteration on top of entity extraction.

    Usage::

        extractor = GleaningExtractor(
            llm_generate_fn=my_llm.agenerate,
            config=GleaningConfig(max_rounds=1, use_json_mode=True),
        )
        result = await extractor.extract_with_gleaning(chunk_text)

    The returned ``ExtractionResult`` contains merged entities/relationships
    from initial + gleaning passes.
    """

    def __init__(
        self,
        llm_generate_fn,  # Callable[[List[Dict], Dict], str] — must accept messages + kwargs
        *,
        config: Optional[GleaningConfig] = None,
        token_counter: Any = None,  # Optional TokenCounter instance for guard
        max_extract_tokens: int = 6000,  # Token safety limit per LLM call
    ) -> None:
        self._llm_call = llm_generate_fn
        self.config = config or GleaningConfig()
        self._token_counter = token_counter
        self._max_extract_tokens = max_extract_tokens

    async def extract_with_gleaning(
        self,
        chunk_text: str,
        *,
        extraction_system_prompt: str = "",
        extraction_user_prompt: str = "",
    ) -> ExtractionResult:
        """Run initial extraction + optional gleaning rounds.

        Parameters
        ----------
        chunk_text : str
            The document chunk text to extract entities from.
        extraction_system_prompt : str
            System prompt for the initial extraction LLM call.
        extraction_user_prompt : str
            User prompt template; ``{chunk_text}`` will be replaced.

        Returns
        -------
        ExtractionResult with merged entities and relationships.
        """
        # --- Step 1: Initial Extraction ---
        user_prompt = extraction_user_prompt.replace("{chunk_text}", chunk_text)
        messages = []
        if extraction_system_prompt:
            messages.append({"role": "system", "content": extraction_system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        t0 = time.monotonic()
        initial_raw = await self._call_llm(messages)
        t1 = time.monotonic()

        initial_entities, initial_rels = self._parse_result(initial_raw)
        initial_result = ExtractionResult(
            entities=initial_entities,
            relationships=initial_rels,
            raw_text=initial_raw,
            tokens_used=self._count_tokens(messages),
            duration_ms=(t1 - t0) * 1000,
            is_gleaning=False,
        )

        log.info(
            "Initial extraction: %d entities, %d relations, %.1fms",
            len(initial_result.entities),
            len(initial_result.relationships),
            initial_result.duration_ms,
        )

        # --- Step 2: Decide if Gleaning is Needed ---
        if not self.config.enabled or self.config.max_rounds <= 0:
            return initial_result

        # Check token budget for gleaning
        glean_prompt = self._build_gleaning_prompt(initial_raw)
        estimated_glean_tokens = self._count_text_tokens(glean_prompt)
        if estimated_glean_tokens > self._max_extract_tokens:
            log.warning(
                "Skipping gleaning: estimated tokens (%d) exceeds limit (%d)",
                estimated_glean_tokens,
                self._max_extract_tokens,
            )
            return initial_result

        # --- Step 3: Gleaning Rounds ---
        current_entities = list(initial_result.entities)
        current_rels = list(initial_result.relationships)

        for round_num in range(1, self.config.max_rounds + 1):
            glean_messages = [
                {"role": "system", "content": extraction_system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": initial_raw},
                {"role": "user", "content": glean_prompt},
            ]

            t2 = time.monotonic()
            glean_raw = await self._call_llm(glean_messages)
            t3 = time.monotonic()

            glean_entities, glean_rels = self._parse_result(glean_raw)
            glean_result = ExtractionResult(
                entities=glean_entities,
                relationships=glean_rels,
                raw_text=glean_raw,
                tokens_used=self._count_tokens(glean_messages),
                duration_ms=(t3 - t2) * 1000,
                is_gleaning=True,
            )

            log.info(
                "Gleaning round %d: %d new entities, %d new relations, %.1fms",
                round_num,
                len(glean_result.entities),
                len(glean_result.relationships),
                glean_result.duration_ms,
            )

            # --- Step 4: Merge (by description length preference) ---
            current_entities = self._merge_entities(current_entities, glean_entities)
            current_rels = self._merge_relationships(current_rels, glean_rels)

        final = ExtractionResult(
            entities=current_entities,
            relationships=current_rels,
            raw_text="",  # Merged — no single raw text
            tokens_used=initial_result.tokens_used
            + getattr(glean_result, "tokens_used", 0),
            duration_ms=initial_result.duration_ms
            + getattr(glean_result, "duration_ms", 0),
            is_gleaning=False,
        )

        log.info(
            "Final after gleaning: %d entities (+%d), %d relations (+%d)",
            len(final.entities), len(final.entities) - len(initial_entities),
            len(final.relationships), len(final.relationships) - len(initial_rels),
        )
        return final

    # -- LLM calling --------------------------------------------------------

    async def _call_llm(self, messages: List[Dict[str, Any]]) -> str:
        """Invoke the LLM generate function."""
        try:
            if hasattr(self._llm_call, "__call__"):
                # Async callable
                if hasattr(self._llm_call, "agenerate"):
                    result = await self._llm_call.agenerate(messages=messages)
                else:
                    result = await self._llm_call(messages)
                return result
            else:
                raise RuntimeError("LLM generate fn is not callable")
        except Exception as e:
            log.error("LLM call failed in gleaning: %s", e)
            return '{"entities": [], "relationships": []}'

    # -- Parsing ------------------------------------------------------------

    def _parse_result(self, raw: str) -> Tuple[List[Dict], List[Dict]]:
        """Parse LLM output into entities and relationships."""
        if self.config.use_json_mode:
            return self._parse_json(raw)
        return self._parse_delimiter(raw)

    def _parse_json(self, raw: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract JSON from potentially markdown-fenced output."""
        text = raw.strip()
        # Strip markdown fences
        if text.startswith("```"):
            first_nl = text.find("\n")
            if first_nl >= 0:
                text = text[first_nl + 1:]
            if text.endswith("```"):
                text = text[:-3].strip()
            elif text.endswith("```\n"):
                text = text[:-4].strip()

        # Try find JSON object
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                obj = json.loads(json_match.group())
                entities = obj.get("entities", [])
                relationships = obj.get("relationships", [])
                return (
                    [e if isinstance(e, dict) else {"name": str(e)} for e in entities],
                    [r if isinstance(r, dict) else {} for r in relationships],
                )
            except json.JSONDecodeError:
                log.warning("Failed to parse JSON from LLM output")

        return [], []

    def _parse_delimiter(self, raw: str) -> Tuple[List[Dict], List[Dict]]:
        """Parse LightRAG-style delimiter-separated format."""
        entities = []
        rels = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line or line == self.config.completion_delimiter:
                continue
            if line.lower().startswith("entity:"):
                parts = line[len("entity:"):].split(",")
                if parts:
                    entities.append({"name": parts[0].strip(), "raw": line})
            elif line.lower().startswith("relation:"):
                parts = line[len("relation:"):].split(",")
                if len(parts) >= 2:
                    rels.append({"source": parts[0].strip(), "target": parts[-1].strip(), "raw": line})
        return entities, rels

    # -- Merging strategies --------------------------------------------------

    @staticmethod
    def _merge_entities(
        base: List[Dict], gleaned: List[Dict]
    ) -> List[Dict]:
        """Merge gleaning entities into base, preferring longer descriptions."""
        name_to_idx: Dict[str, int] = {}
        for i, e in enumerate(base):
            key = e.get("name") or e.get("entity_name", "")
            if key:
                name_to_idx[key] = i

        for ge in gleaned:
            gname = ge.get("name") or ge.get("entity_name", "")
            if not gname:
                continue
            if gname in name_to_idx:
                idx = name_to_idx[gname]
                existing_desc = (
                    base[idx].get("description") or base[idx].get("desc", "") or ""
                )
                glean_desc = ge.get("description") or ge.get("desc", "") or ""
                # Keep the version with more detailed description
                if len(glean_desc) > len(existing_desc):
                    base[idx] = ge
            else:
                base.append(ge)
                name_to_idx[gname] = len(base) - 1
        return base

    @staticmethod
    def _merge_relationships(
        base: List[Dict], gleaned: List[Dict]
    ) -> List[Dict]:
        """Merge gleaning relationships into base, preferring longer descriptions."""
        # Use source+target as dedup key
        pair_to_idx: Dict[str, int] = {}
        for i, r in enumerate(base):
            src = r.get("source") or r.get("src_id", "")
            tgt = r.get("target") or r.get("tgt_id", "")
            pair_to_idx[f"{src}->{tgt}"] = i

        for gr in gleaned:
            src = gr.get("source") or gr.get("src_id", "")
            tgt = gr.get("target") or gr.get("tgt_id", "")
            key = f"{src}->{tgt}"
            if key in pair_to_idx:
                idx = pair_to_idx[key]
                existing_desc = base[idx].get("description") or ""
                glean_desc = gr.get("description") or ""
                if len(glean_desc) > len(existing_desc):
                    base[idx] = gr
            else:
                base.append(gr)
                pair_to_idx[key] = len(base) - 1
        return base

    # -- Helpers -------------------------------------------------------------

    def _build_gleaning_prompt(self, previous_response: str) -> str:
        """Build the gleaning follow-up prompt."""
        tmpl = GLEANING_JSON_PROMPT if self.config.use_json_mode else GLEANING_TEXT_PROMPT
        return tmpl.format(
            max_total_records=self.config.max_total_records_per_round,
            max_entity_records=self.config.max_entity_records_per_round,
            language=self.config.language,
            completion_delimiter=self.config.completion_delimiter,
        )

    def _count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count approximate tokens in a message list."""
        if self._token_counter:
            return self._token_counter.count_messages(messages).num_tokens
        return sum(len(str(m)) // 3 for m in messages)

    def _count_text_tokens(self, text: str) -> int:
        if self._token_counter:
            return self._token_counter.count(text).num_tokens
        return len(text) // 3
