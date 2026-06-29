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
Intelligent memory lifecycle components (PowerMem-style).

- ImportanceEvaluator: score memory importance (LLM-based or heuristic)
- EbbinghausDecay: retention curve with access reinforcement
- MemoryOptimizer: deduplication and conflict detection
- EntityExtractor: extract query entities for graph-centric retrieval boost
"""

import hashlib
import json
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from hugegraph_llm.config.memory_config import memory_settings
from hugegraph_llm.utils.log import log


class ImportanceEvaluator:
    """Score memory importance in [0, 1].

    PowerMem uses LLM-based importance evaluation (see
    src/powermem/intelligence/importance_evaluator.py). We provide a
    lightweight heuristic by default plus an optional LLM callback.
    """

    def __init__(
        self,
        llm_callback: Optional[Callable[[str], str]] = None,
        use_llm: bool = False,
    ):
        self.llm_callback = llm_callback
        self.use_llm = use_llm and llm_callback is not None

    def score(self, content: str) -> float:
        if not content:
            return 0.0

        if self.use_llm:
            try:
                return self._llm_score(content)
            except Exception as e:
                log.warning("LLM importance scoring failed: %s; falling back to heuristic", e)

        return self._heuristic_score(content)

    def _heuristic_score(self, content: str) -> float:
        """Heuristic based on information density, entities, and actionability."""
        score = 0.5

        # Prefer concrete facts (numbers, dates, named entities) over chitchat
        indicators = [
            (r"\d{4}[-/年]\d{1,2}[-/月]", 0.15),  # dates
            (r"\d+", 0.05),  # numbers
            (r"[A-Z][a-zA-Z ]{2,20}", 0.05),  # capitalized names (Latin)
            (r"[\u4e00-\u9fa5]{2,6}(?:公司|集团|学校|医院|银行|厂)", 0.1),  # orgs
            (r"喜欢|讨厌|擅长|目标|计划|负责|项目|客户|同事|朋友", 0.1),  # preferences/relations
        ]
        for pat, weight in indicators:
            if re.search(pat, content):
                score = min(1.0, score + weight)

        # Penalize very short / very long noise
        length = len(content)
        if length < 10:
            score *= 0.7
        elif length > 500:
            score *= 0.9

        return round(score, 4)

    def _llm_score(self, content: str) -> float:
        prompt = (
            "Evaluate the importance of the following memory on a scale of 0 to 1. "
            "0 = trivial / ephemeral, 1 = critical / highly reusable. "
            "Return only a JSON object with a single key 'score' and a float value.\n\n"
            f"Memory: {content}\n"
        )
        raw = self.llm_callback(prompt)  # type: ignore[misc]
        try:
            data = json.loads(raw)
            return float(data.get("score", 0.5))
        except Exception:
            # Try to extract the first number in [0,1]
            m = re.search(r"0?\.\d+|1\.0|1|0", raw)
            if m:
                return min(1.0, max(0.0, float(m.group(0))))
            return 0.5


class EbbinghausDecay:
    """Ebbinghaus forgetting curve with access reinforcement."""

    def __init__(self, k: Optional[float] = None, reinforce: Optional[float] = None):
        self.k = k if k is not None else memory_settings.ebbinghaus_k
        self.reinforce = reinforce if reinforce is not None else memory_settings.ebbinghaus_reinforce

    def retention(
        self,
        initial_score: float,
        elapsed_hours: float,
        access_count: int = 0,
    ) -> float:
        ret = initial_score * math.exp(-self.k * elapsed_hours)
        ret = min(1.0, max(0.0, ret + access_count * self.reinforce))
        return round(ret, 4)

    def time_to_rehearsal(self, retention: float, threshold: float = 0.3) -> float:
        """Estimate hours until retention falls below threshold."""
        if retention <= threshold:
            return 0.0
        return max(0.0, -math.log(threshold / retention) / self.k)


class MemoryOptimizer:
    """Deduplication and conflict detection (PowerMem-style)."""

    def __init__(self, similarity_threshold: float = 0.92):
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def deduplicate(
        self,
        memories: List[Dict[str, Any]],
        strategy: str = "exact",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return (kept, duplicates)."""
        if strategy == "exact":
            seen: set = set()
            kept, duplicates = [], []
            for mem in memories:
                h = self.content_hash(mem.get("content", ""))
                if h in seen:
                    duplicates.append(mem)
                else:
                    seen.add(h)
                    kept.append(mem)
            return kept, duplicates

        if strategy == "semantic":
            # Lightweight semantic dedup using normalized content equality
            kept, duplicates = [], []
            for mem in memories:
                norm = self._normalize(mem.get("content", ""))
                if any(norm == self._normalize(k.get("content", "")) for k in kept):
                    duplicates.append(mem)
                else:
                    kept.append(mem)
            return kept, duplicates

        raise ValueError(f"Unknown dedup strategy: {strategy}")

    def detect_conflict(
        self,
        new_fact: str,
        existing_facts: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Detect contradictory facts using a simple heuristic.

        PowerMem uses LLM prompts for contradiction detection (see
        src/powermem/prompts/intelligent_memory_prompts.py). Here we provide a
        rule-based fallback for common negation patterns.
        """
        negations = {"不", "没", "无", "非", "not", "no", "never"}
        for fact in existing_facts:
            if not fact:
                continue
            new_norm = self._normalize(new_fact)
            old_norm = self._normalize(fact)
            # If one is a negated version of the other, flag conflict
            new_has_neg = any(n in new_norm for n in negations)
            old_has_neg = any(n in old_norm for n in negations)
            if new_has_neg != old_has_neg:
                common = re.sub(r"[不没非无notno\s]+", "", new_norm)
                if common and common in old_norm:
                    return {
                        "type": "contradiction",
                        "existing": fact,
                        "new": new_fact,
                    }
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text.lower()).strip()


class EntityExtractor:
    """Extract candidate entities from query text for graph-centric boosting."""

    def __init__(self, llm_callback: Optional[Callable[[str], str]] = None):
        self.llm_callback = llm_callback

    def extract(self, text: str) -> List[Dict[str, str]]:
        """Return list of {'name': ..., 'type': ...} candidates."""
        entities = self._rule_based(text)
        if self.llm_callback:
            try:
                llm_entities = self._llm_extract(text)
                entities = self._merge(entities, llm_entities)
            except Exception as e:
                log.warning("LLM entity extraction failed: %s", e)
        return entities

    def _rule_based(self, text: str) -> List[Dict[str, str]]:
        """Chinese and English named entity heuristics."""
        entities = []
        # Chinese person/org/location patterns
        for m in re.finditer(
            r"([\u4e00-\u9fa5]{2,6})(?:公司|集团|学校|银行|医院|厂|团队|部门)", text
        ):
            entities.append({"name": m.group(0), "type": "organization"})
        for m in re.finditer(
            r"(?:我(?:的|叫))([\u4e00-\u9fa5]{2,4})|(?:他叫|她叫|同事|朋友)([\u4e00-\u9fa5]{2,4})",
            text,
        ):
            name = m.group(1) or m.group(2)
            if name:
                entities.append({"name": name, "type": "person"})
        # English capitalized multi-word names
        for m in re.finditer(r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+", text):
            entities.append({"name": m.group(0), "type": "person"})
        # Locations
        for m in re.finditer(
            r"([\u4e00-\u9fa5]{2,6})(?:市|省|区|县|路|街|楼|层|国)", text
        ):
            entities.append({"name": m.group(0), "type": "location"})
        return entities

    def _llm_extract(self, text: str) -> List[Dict[str, str]]:
        prompt = (
            "Extract named entities from the following query. "
            "Return only a JSON object with key 'entities' containing a list of "
            "{\"name\": \"...\", \"type\": \"person|organization|location|skill|concept\"}.\n\n"
            f"Query: {text}\n"
        )
        raw = self.llm_callback(prompt)  # type: ignore[misc]
        try:
            data = json.loads(raw)
            return data.get("entities", [])
        except Exception:
            return []

    @staticmethod
    def _merge(
        a: List[Dict[str, str]],
        b: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        names = set()
        merged = []
        for e in a + b:
            key = (e.get("name", ""), e.get("type", ""))
            if key[0] and key not in names:
                names.add(key)
                merged.append(e)
        return merged
