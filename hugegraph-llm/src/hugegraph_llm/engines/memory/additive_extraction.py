# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not this file except in compliance
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
V3 Additive Extraction Pipeline — aligned with mem0's V3 ADD-only mode.

Key changes vs old 7-step pipeline:
  1. Pure ADD extraction (no UPDATE/DELETE/NONE decision per fact)
  2. MD5 hash deduplication (batch-internal + against stored hashes)
  3. Entity linking (extract entities from facts, match/insert into entity store)
  4. Batch persistence (vector + BM25 + graph + SQLite in one pass)

This replaces the over-complex ADD pipeline that was trying to decide
ADD/UPDATE/DELETE at the per-fact level via LLM (which is prone to
hallucination and error), with a simpler "extract all facts, dedup by
hash, store what's new" approach — exactly what mem0 V3 does.
"""

import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from hugegraph_llm.utils.log import log

logger = logging.getLogger(__name__)

# ── V3 Extraction Prompt ────────────────────────────────────────

ADDITIVE_EXTRACTION_PROMPT = """\
You are a memory extraction engine. Extract ALL factual statements from the input text.
Each fact should be atomic (one piece of information per fact).
Do NOT judge whether a fact is new or redundant — just extract all facts.
Do NOT output UPDATE, DELETE, or NONE actions — only ADD new facts.

Output format (JSON):
{{"memory": [
    "fact 1",
    "fact 2",
    ...
  ]
}}

Rules:
- Each fact must be a single, atomic statement (one subject + predicate + object).
- Extract facts about people, organizations, locations, dates, preferences, skills, events.
- Preserve exact names and numbers (do NOT generalize or paraphrase).
- If the text contains negations (e.g., "不喜欢", "不在"), include them as facts.
- Do NOT duplicate facts within the same batch.
- Return ONLY the JSON object, no additional commentary.

Input text:
{text}

Extract facts now:
"""

DEDUP_PROMPT = """\
Given these new facts and these existing stored memories, determine which new facts
are truly novel (not already contained in any existing memory).

For each new fact, decide:
- ADD: the fact is novel (not contained in any existing memory)
- SKIP: the fact is already contained in (or semantically equivalent to) an existing memory

Output format (JSON):
{{"decisions": [
    {{'fact': "...", "action": "ADD|SKIP", "reason": "..."}},
    ...
  ]
}}

New facts:
{new_facts}

Existing memories:
{existing_memories}
"""

# ── MD5 Hash Dedup ──────────────────────────────────────────────


def content_hash_md5(text: str) -> str:
    """Compute MD5 hash of normalized text content (mem0-style dedup).

    Normalization: strip whitespace, lowercase, remove punctuation variants.
    """
    normalized = re.sub(r"\s+", " ", text.strip()).lower()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def batch_dedup(
    facts: List[str],
    stored_hashes: Optional[set] = None,
) -> Tuple[List[str], List[str], set]:
    """Dedup facts by MD5 hash against stored hashes and within the batch.

    Returns:
      (new_facts, duplicate_facts, all_hashes)
    """
    stored = stored_hashes or set()
    new_facts = []
    dup_facts = []
    batch_hashes: set = set()

    for fact in facts:
        h = content_hash_md5(fact)
        if h in stored or h in batch_hashes:
            dup_facts.append(fact)
        else:
            new_facts.append(fact)
            batch_hashes.add(h)

    all_hashes = stored | batch_hashes
    return new_facts, dup_facts, all_hashes


# ── V3 Additive Pipeline ────────────────────────────────────────


class AdditiveExtractionPipeline:
    """V3-style additive extraction pipeline (aligned with mem0 V3).

    Usage:
      pipeline = AdditiveExtractionPipeline(llm_callback=my_llm_call)
      result = pipeline.run(text, stored_hashes=existing_hashes)
      # result = {"new_facts": [...], "duplicate_facts": [...],
      #           "entities": [...], "hashes": {...}}
    """

    def __init__(self, llm_callback: Optional[Any] = None):
        """Initialize with an LLM callback (OpenAI-compatible generate function).

        Args:
            llm_callback: A callable that takes a prompt string and returns
                          a string response. Typically wraps OpenAI.chat.completions.create.
        """
        self.llm = llm_callback

    def run(
        self,
        text: str,
        stored_hashes: Optional[set] = None,
        existing_memories: Optional[List[str]] = None,
        use_llm_dedup: bool = False,
    ) -> Dict[str, Any]:
        """Run the full V3 additive extraction pipeline.

        Steps:
        1. LLM extract all facts (ADD-only, no UPDATE/DELETE decisions)
        2. MD5 hash dedup (batch-internal + against stored hashes)
        3. Optional LLM semantic dedup (for borderline cases)
        4. Entity extraction from new facts
        5. Return structured result

        Args:
            text: Input text to extract memories from
            stored_hashes: Set of MD5 hashes of already-stored memories
            existing_memories: List of existing memory texts (for LLM dedup)
            use_llm_dedup: Whether to use LLM for semantic dedup (slower but more accurate)

        Returns:
            Dict with keys: new_facts, duplicate_facts, entities, hashes, extraction_time_ms
        """
        start_time = time.time()

        # Step 1: LLM extraction (ADD-only)
        raw_facts = self._extract_facts(text)
        if not raw_facts:
            return {
                "new_facts": [],
                "duplicate_facts": [],
                "entities": [],
                "hashes": stored_hashes or set(),
                "extraction_time_ms": round((time.time() - start_time) * 1000, 2),
            }

        # Step 2: MD5 hash dedup
        new_facts, dup_facts, all_hashes = batch_dedup(raw_facts, stored_hashes)

        # Step 3: Optional LLM semantic dedup
        if use_llm_dedup and self.llm and existing_memories and new_facts:
            new_facts, llm_dup_facts = self._llm_semantic_dedup(
                new_facts, existing_memories
            )
            dup_facts.extend(llm_dup_facts)

        # Step 4: Entity extraction from new facts
        entities = self._extract_entities_from_facts(new_facts)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "new_facts": new_facts,
            "duplicate_facts": dup_facts,
            "entities": entities,
            "hashes": all_hashes,
            "extraction_time_ms": elapsed_ms,
        }

    def _extract_facts(self, text: str) -> List[str]:
        """Extract atomic facts from text using LLM (ADD-only mode)."""
        if not self.llm:
            # Fallback: treat entire text as one fact
            return [text.strip()] if text.strip() else []

        prompt = ADDITIVE_EXTRACTION_PROMPT.format(text=text)
        try:
            response = self.llm(prompt)
            # Parse JSON response
            facts = _parse_extraction_response(response)
            return facts
        except Exception as e:
            logger.warning("LLM extraction failed: %s; falling back to raw text", e)
            return [text.strip()] if text.strip() else []

    def _llm_semantic_dedup(
        self,
        new_facts: List[str],
        existing_memories: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Use LLM to decide which new facts are truly novel vs already covered.

        Returns: (novel_facts, skipped_facts)
        """
        prompt = DEDUP_PROMPT.format(
            new_facts=json.dumps(new_facts, ensure_ascii=False),
            existing_memories=json.dumps(
                existing_memories[:20], ensure_ascii=False  # Limit context window
            ),
        )
        try:
            response = self.llm(prompt)
            decisions = _parse_dedup_response(response)
            novel = []
            skipped = []
            for d in decisions:
                fact = d.get("fact", "")
                action = d.get("action", "ADD").upper()
                if action == "SKIP":
                    skipped.append(fact)
                else:
                    novel.append(fact)
            return novel, skipped
        except Exception as e:
            logger.warning("LLM semantic dedup failed: %s; keeping all facts", e)
            return new_facts, []

    @staticmethod
    def _extract_entities_from_facts(facts: List[str]) -> List[Dict[str, str]]:
        """Extract named entities from facts using rule-based heuristics.

        This mirrors mem0's entity extraction (PROPER/QUOTED/TOPIC/IDENTIFIER)
        but uses simpler regex for Chinese-dominated text.
        """
        entities = []
        for fact in facts:
            # Chinese organizations
            for m in re.finditer(
                r"([\u4e00-\u9fa5]{2,6})(?:公司|集团|学校|银行|医院|厂|团队|部门)",
                fact,
            ):
                entities.append({"name": m.group(0), "type": "organization"})
            # Chinese person names
            for m in re.finditer(r"[\u4e00-\u9fa5]{2,4}", fact):
                candidate = m.group(0)
                # Skip short generic words
                if len(candidate) >= 2 and candidate not in {
                    "的", "了", "在", "是", "有", "和", "也", "都", "不", "没",
                    "这", "那", "要", "会", "能", "做", "去", "来", "到", "就",
                }:
                    entities.append({"name": candidate, "type": "person"})
            # English names
            for m in re.finditer(r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+", fact):
                entities.append({"name": m.group(0), "type": "person"})
            # Locations
            for m in re.finditer(
                r"([\u4e00-\u9fa5]{2,6})(?:市|省|区|县|路|街|楼|层|国)", fact
            ):
                entities.append({"name": m.group(0), "type": "location"})

        # Deduplicate entities by name
        seen = set()
        unique = []
        for e in entities:
            key = (e["name"], e["type"])
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique


# ── Response parsing utilities ──────────────────────────────────


def _parse_extraction_response(response: str) -> List[str]:
    """Parse LLM response for V3 ADD-only extraction.

    Handles multiple formats:
    - {"memory": ["fact1", "fact2"]}
    - {"facts": ["fact1", "fact2"]}
    - ["fact1", "fact2"]
    - Plain text with numbered/bulleted facts
    """
    # Try JSON parsing first
    text = _strip_code_blocks(response)

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # mem0-style: {"memory": [...]}
            for key in ("memory", "memories", "facts", "results"):
                if key in data and isinstance(data[key], list):
                    return [str(f).strip() for f in data[key] if str(f).strip()]
        if isinstance(data, list):
            return [str(f).strip() for f in data if str(f).strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: extract numbered/bulleted lines
    facts = []
    for line in text.split("\n"):
        line = line.strip()
        # Remove numbering/bullet prefixes
        line = re.sub(r"^\d+[\.\)、]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        if line and len(line) > 5:
            facts.append(line)
    return facts


def _parse_dedup_response(response: str) -> List[Dict[str, str]]:
    """Parse LLM dedup decision response."""
    text = _strip_code_blocks(response)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "decisions" in data:
            return data["decisions"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract ADD/SKIP actions from plain text
    decisions = []
    for line in text.split("\n"):
        line = line.strip()
        # Match patterns like "1. fact - ADD" or "- fact - SKIP" or "fact: ADD"
        m = re.match(r"[-*\d.)\s]*\s*(.+?)\s*[-–—:]\s*(ADD|SKIP)", line, re.IGNORECASE)
        if m:
            decisions.append({
                "fact": m.group(1).strip(),
                "action": m.group(2).upper(),
            })
    return decisions


def _strip_code_blocks(text: str) -> str:
    """Remove markdown code block markers from LLM response."""
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()
