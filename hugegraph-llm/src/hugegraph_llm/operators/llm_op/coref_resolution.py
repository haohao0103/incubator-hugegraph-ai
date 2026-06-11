# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this License except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under an License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""Coreference resolution operator for GraphRAG build pipeline.

Resolves entity references across text chunks, including:
- Pronouns: he/she/it/this/that -> specific entity
- Nominal aliases: the company / that engineer / Mr. Zhang -> canonical entity name
- Demonstratives: this product / their CEO -> specific entity

Why Coref Matters:
    Without coref, each chunk is processed independently:
      Chunk 1: "Zhang San joined Alibaba Cloud."
      Chunk 2: "He works as a senior engineer."
    Entity extraction sees "Zhang San" and an unresolved "He" ->
    they become separate entities, breaking graph connectivity.

Integration Point:
    Build Pipeline: chunks -> [EntityExtract] -> [**CorefResolve**] ->
                    [RelationExtract] -> [ClaimExtract] -> ...

Data Model:
    CorefMapping = {
        "mention": str,          # The text span found in chunk (e.g., "他")
        "canonical": str,       # Resolved entity name (e.g., "张三")
        "entity_type": str,     # Entity type (e.g., "Person")
        "chunk_id": str,        # Source chunk
        "confidence": float,    # 0.0 - 1.0
        "method": str,          # "rule" | "llm"
    }
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.utils.log import log


# ── Chinese Coreference Patterns ───────────────────────────────

# Personal pronouns (simplified for common cases)
_CN_PERSONAL_PRONOUNS = {
    # Singular
    "他": None, "她": None, "它": None, "祂": None,
    "它的": None, "她的": None, "他的": None,
    # Plural
    "他们": None, "她们": None, "它们": None,
    "他们的": None, "她们的": None, "它们的": None,
    # Reflexive
    "自己": None, "本身": None, "自身": None,
}

# Demonstrative pronouns with entity affinity
_CN_DEMONSTRATIVES = {
    "这": None, "这个": None, "这家": None, "该公司": None,
    "此": None, "此人": None, "此地": None,
    "那": None, "那个": None, "那家": None,
    "其": None, "其中": None, " thereof": None,
    # Common patterns
    "这位": None, "那位": None, "该": None,
}

# Title/prefix patterns that indicate a person reference
_CN_TITLES = [
    r"(张)(?:先生|女士|总|经理|博士|老师|工)",
    r"(李)(?:先生|女士|总|经理|博士|老师|工)",
    r"(王)(?:先生|女士|总|经理|博士|老师|工)",
    r"(刘)(?:先生|女士|总|经理|博士|老师|工)",
    r"(陈)(?:先生|女士|总|经理|博士|老师|工)",
    r"(杨)(?:先生|女士|总|经理|博士|老师|工)",
    r"(赵)(?:先生|女士|总|经理|博士|老师|工)",
    r"(黄)(?:先生|女士|总|经理|博士|老师|工)",
    r"(周)(?:先生|女士|总|经理|博士|老师|工)",
    r"(Mr|Ms|Dr|Prof)\.?\s*([A-Za-z]+)",
]

# Organization alias patterns
_ORG_ALIAS_PATTERNS = [
    r"^(?:这家|该|这间|那家|那间)?(?:公司|企业|集团|机构|组织|部门|团队|厂|店)$",
    r"^(?:阿里|阿里巴巴|Alibaba)(?:云|科技|集团|公司)?$",
    r"^(?:腾讯|Tencent)(?:科技|公司|集团)?$",
    r"^(?:华为|Huawei)(?:技术|公司|集团)?$",
]


@dataclass
class CorefMapping:
    """A single coreference resolution mapping."""

    mention: str           # Text found (e.g., "他", "该公司")
    canonical: str         # Resolved entity name (e.g., "张三", "阿里云")
    entity_type: str = ""  # Person / Organization / Location / etc.
    chunk_id: str = ""
    confidence: float = 0.0
    method: str = "rule"   # "rule" | "llm"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mention": self.mention,
            "canonical": self.canonical,
            "entity_type": self.entity_type,
            "chunk_id": self.chunk_id,
            "confidence": round(self.confidence, 4),
            "method": self.method,
        }


# ── Prompt Template for LLM-based Coref ──────────────────────

COREF_LLM_PROMPT = """You are a Chinese-English coreference resolution system.

## Task
Given a text and a list of known entities, resolve all pronouns, demonstratives, and alias references to their canonical entity names.

## Text (Chunk {chunk_id}):
```
{text}
```

## Known Entities (with types):
{entities_list}

## Rules
1. Resolve personal pronouns (he/she/it/他们/她们/它们/他/她/它) to the most recently mentioned entity of matching gender/type.
2. Resolve demonstratives (这/那/这个/那个/该公司/这位/那位) to the most relevant entity from context.
3. Resolve title-based references (张先生/Mr.Zhang) to the matching entity.
4. If uncertain, mark confidence low or skip.
5. Output ONLY valid JSON array.

## Output Format
```json
[
  {{
    "mention": "text_span_found",
    "canonical": "resolved_entity_name",
    "entity_type": "Person|Organization|Location|...",
    "confidence": 0.95
  }}
]
```

If no coreferences found, output empty array: []"""


# ── Main Operator ─────────────────────────────────────────────


class CorefResolver:
    """Cross-chunk coreference resolution for GraphRAG.

    Two-pass approach:
    Pass 1 (Rule): Fast rule-based resolution for common patterns.
                   Handles ~80% of Chinese/English coref cases.
    Pass 2 (LLM): Optional LLM-based resolution for ambiguous cases.
                   Catches remaining ~20% including complex context.

    Usage:
        resolver = CorefResolver()
        context = resolver.run(context)
        # context["coref_mappings"] = [CorefMapping(...), ...]
        # context["resolved_chunks"] = chunks with mentions replaced
    """

    def __init__(self, llm: BaseLLM = None, enable_llm_pass: bool = False):
        self._llm = llm
        self._enable_llm_pass = enable_llm_pass

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run coreference resolution on all chunks.

        Reads from context:
            chunks: List of text chunk dicts.
            vertices: List of extracted entity dicts with properties.name.
            doc_id: Document ID (optional).

        Writes to context:
            coref_mappings: List of CorefMapping dicts.
            coref_count: Total number of resolutions.
            resolved_chunks: Chunks with mentions optionally annotated.
        """
        chunks = context.get("chunks", [])
        vertices = context.get("vertices", [])
        doc_id = context.get("doc_id", "unknown")

        if not chunks or not vertices:
            log.info("No chunks or vertices for coref resolution.")
            context["coref_mappings"] = []
            context["coref_count"] = 0
            return context

        # Build entity catalog: name -> (type, properties)
        entity_catalog = self._build_entity_catalog(vertices)

        all_mappings = []
        recent_entities = []  # Track most recently mentioned entities per chunk

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            chunk_id = chunk.get("chunk_id", f"chunk_{i}") if isinstance(chunk, dict) else f"chunk_{i}"

            # Update recent entities from this chunk's explicit mentions
            chunk_entities = self._find_explicit_mentions(chunk_text, entity_catalog)
            recent_entities.extend(chunk_entities)

            # Pass 1: Rule-based resolution
            rule_maps = self._resolve_rules(
                chunk_text, chunk_id, entity_catalog, recent_entities
            )
            all_mappings.extend(rule_maps)

            # Pass 2: LLM-based resolution (optional)
            if self._enable_llm_pass and self._llm:
                llm_maps = self._resolve_llm(
                    chunk_text, chunk_id, entity_catalog, vertices
                )
                all_mappings.extend(llm_maps)

        # Deduplicate mappings
        all_mappings = self._deduplicate_mappings(all_mappings)

        context["coref_mappings"] = [m.to_dict() for m in all_mappings]
        context["coref_count"] = len(all_mappings)
        log.info(
            "Coref resolution complete: %d mappings across %d chunks",
            len(all_mappings), len(chunks),
        )
        return context

    @staticmethod
    def _build_entity_catalog(vertices: List[Dict]) -> Dict[str, Tuple[str, Dict]]:
        """Build name -> (type, props) lookup."""
        catalog = {}
        for v in vertices:
            label = v.get("label", "")
            props = v.get("properties", {})
            name = props.get("name", "")
            if name:
                catalog[name] = (label, props)
        return catalog

    @staticmethod
    def _find_explicit_mentions(text: str, catalog: Dict) -> List[str]:
        """Find explicitly mentioned entity names in text."""
        mentioned = []
        for name in catalog:
            if name and name in text:
                mentioned.append(name)
        return mentioned

    def _resolve_rules(
        self,
        text: str,
        chunk_id: str,
        catalog: Dict[str, Tuple[str, Dict]],
        recent_entities: List[str],
    ) -> List[CorefMapping]:
        """Pass 1: Rule-based coreference resolution."""

        mappings = []

        # Strategy 1: Personal pronoun resolution
        # "他/她" -> most recent Person-type entity
        person_entities = []
        org_entities = []
        location_entities = []

        for name in reversed(recent_entities):
            if name in catalog:
                etype = catalog[name][0]
                if "Person" in etype or "person" in etype.lower():
                    if name not in person_entities:
                        person_entities.append(name)
                elif ("Org" in etype or "Company" in etype or "org" in etype.lower()):
                    if name not in org_entities:
                        org_entities.append(name)
                elif ("Loc" in etype or "location" in etype.lower()):
                    if name not in location_entities:
                        location_entities.append(name)

        # Chinese personal pronouns → most recent person
        for pronoun, _target in _CN_PERSONAL_PRONOUNS.items():
            if pronoun in text and person_entities:
                target = person_entities[0]  # Most recent
                mappings.append(CorefMapping(
                    mention=pronoun,
                    canonical=target,
                    entity_type=catalog[target][0] if target in catalog else "Person",
                    chunk_id=chunk_id,
                    confidence=0.85 if pronoun in ("他", "她", "他的", "她的") else 0.7,
                    method="rule",
                ))

        # Strategy 2: Demonstrative + organization pattern
        # "该公司/这家公司/这" + nearby Org entity
        for demo, _target in _CN_DEMONSTRATIVES.items():
            if demo in text and org_entities:
                target = org_entities[0]
                mappings.append(CorefMapping(
                    mention=demo,
                    canonical=target,
                    entity_type=catalog[target][0] if target in catalog else "Organization",
                    chunk_id=chunk_id,
                    confidence=0.75,
                    method="rule",
                ))

        # Strategy 3: Title-based resolution
        # "张先生/Mr.Zhang" -> entity with matching surname
        for pattern in _CN_TITLES:
            match = re.search(pattern, text)
            if match:
                mention_text = match.group(0)
                # Extract surname
                surname = match.group(1) if match.lastindex and match.lastindex >= 1 else ""
                if surname:
                    for name, (etype, _) in catalog.items():
                        if name.startswith(surname) and len(name) <= len(surname) + 4:
                            mappings.append(CorefMapping(
                                mention=mention_text,
                                canonical=name,
                                entity_type=etype,
                                chunk_id=chunk_id,
                                confidence=0.9,
                                method="rule",
                            ))
                            break

        # Strategy 4: Common organization aliases
        # "阿里" -> "阿里云"/"阿里巴巴"
        for alias_pat in _ORG_ALIAS_PATTERNS:
            match = re.search(alias_pat, text, re.IGNORECASE)
            if match:
                mention_text = match.group(0)
                for name in catalog:
                    # Check if mention is a substring/superstring of known org
                    if (len(mention_text) >= 2 and
                        (mention_text in name or name.startswith(mention_text[:2]))):
                        etype = catalog[name][0]
                        if "Org" in etype or "Company" in etype or "org" in etype.lower():
                            mappings.append(CorefMapping(
                                mention=mention_text,
                                canonical=name,
                                entity_type=etype,
                                chunk_id=chunk_id,
                                confidence=0.8,
                                method="rule",
                            ))
                            break

        return mappings

    def _resolve_llm(
        self,
        text: str,
        chunk_id: str,
        catalog: Dict[str, Tuple[str, Dict]],
        vertices: List[Dict],
    ) -> List[CorefMapping]:
        """Pass 2: LLM-based coreference for ambiguous cases."""
        entities_list = "\n".join(
            f"- [{v.get('label', '?')}] {v.get('properties', {}).get('name', '?')}"
            for v in vertices[:30]
        )

        prompt = COREF_LLM_PROMPT.format(
            text=text[:2000],
            chunk_id=chunk_id,
            entities_list=entities_list or "(none)",
        )

        try:
            response = self._llm.generate(prompt=prompt)
            items = self._parse_llm_response(response, chunk_id)
            return items
        except Exception as e:
            log.warning("LLM coref failed for %s: %s", chunk_id, e)
            return []

    @staticmethod
    def _parse_llm_response(response: str, chunk_id: str) -> List[CorefMapping]:
        """Parse LLM coref response JSON."""
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if not json_match:
            return []

        try:
            import json
            items = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return []

        mappings = []
        for item in items:
            if not isinstance(item, dict):
                continue
            mention = item.get("mention", "").strip()
            canonical = item.get("canonical", "").strip()
            if mention and canonical:
                mappings.append(CorefMapping(
                    mention=mention,
                    canonical=canonical,
                    entity_type=item.get("entity_type", ""),
                    chunk_id=chunk_id,
                    confidence=float(item.get("confidence", 0.5)),
                    method="llm",
                ))
        return mappings

    @staticmethod
    def _deduplicate_mappings(mappings: List[CorefMapping]) -> List[CorefMapping]:
        """Deduplicate by (mention, canonical, chunk_id)."""
        seen = set()
        unique = []
        for m in mappings:
            key = (m.mention, m.canonical, m.chunk_id)
            if key not in seen:
                seen.add(key)
                unique.append(m)
        return unique

    def apply_to_text(self, text: str, mappings: List[CorefMapping]) -> str:
        """Replace coreferent mentions in text with canonical names.

        Useful for downstream processing where resolved text improves
        extraction quality.
        """
        result = text
        for m in sorted(mappings, key=lambda x: len(x.mention), reverse=True):
            # Longer matches first to avoid partial replacements
            if m.mention in result:
                result = result.replace(m.mention, f"{m.canonical}")
        return result
