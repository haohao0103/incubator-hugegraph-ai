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

"""Claim extraction operator for GraphRAG.

Extracts atomic factual claims from text chunks, following the Microsoft
GraphRAG claim pattern. Claims are finer-grained than entity/relation triples:
they include a veracity status, supporting evidence, and source attribution.

A Claim answers: "What specific fact does this text assert, and how
confident are we that it is true?"

Data Model:
    Claim = {
        "claim_id": str,           # Unique identifier
        "subject": str,            # Primary entity
        "predicate": str,          # Relation / attribute name
        "object": str,             # Value / target entity
        "description": str,        # Natural language description
        "status": str,             # "supporting" | "contradicting" | "not_enough_info"
        "confidence": float,       # 0.0 - 1.0 extraction confidence
        "source_text": str,        # Evidence snippet from original text
        "chunk_id": str,           # Source chunk identifier
        "doc_id": str,             # Source document identifier
        "start_char": int,         # Start offset in chunk
        "end_char": int,           # End offset in chunk
    }

Integration Point:
    Build Pipeline: chunks -> [EntityExtract] -> [RelationExtract] ->
                    [ClaimExtract] -> [EntityResolution] -> Commit2Graph

    Global Search: communities -> collect community claims ->
                   CommunityReport (enriched with claims)
"""

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.utils.log import log


# ── Claim Status ──────────────────────────────────────────────


class ClaimStatus(str, Enum):
    """Veracity status of a extracted claim."""

    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    NOT_ENOUGH_INFO = "not_enough_info"


# ── Claim Data Class ──────────────────────────────────────────


@dataclass
class Claim:
    """An atomic factual assertion extracted from text."""

    claim_id: str = ""
    subject: str = ""
    predicate: str = ""
    object: str = ""
    description: str = ""
    status: ClaimStatus = ClaimStatus.NOT_ENOUGH_INFO
    confidence: float = 0.0
    source_text: str = ""
    chunk_id: str = ""
    doc_id: str = ""
    start_char: int = 0
    end_char: int = 0

    def __post_init__(self):
        if not self.claim_id:
            raw = f"{self.subject}|{self.predicate}|{self.object}|{self.source_text[:50]}"
            self.claim_id = f"claim-{hashlib.md5(raw.encode()).hexdigest()[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "description": self.description,
            "status": self.status.value if isinstance(self.status, ClaimStatus) else self.status,
            "confidence": round(self.confidence, 4),
            "source_text": self.source_text,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Claim":
        raw_status = d.get("status", "not_enough_info")
        try:
            status = ClaimStatus(raw_status)
        except ValueError:
            status = ClaimStatus.NOT_ENOUGH_INFO
        return Claim(
            claim_id=d.get("claim_id", ""),
            subject=d.get("subject", ""),
            predicate=d.get("predicate", ""),
            object=d.get("object", ""),
            description=d.get("description", ""),
            status=status,
            confidence=d.get("confidence", 0.0),
            source_text=d.get("source_text", ""),
            chunk_id=d.get("chunk_id", ""),
            doc_id=d.get("doc_id", ""),
            start_char=d.get("start_char", 0),
            end_char=d.get("end_char", 0),
        )

    def triple(self) -> tuple:
        """Return as (subject, predicate, object) tuple for graph storage."""
        return (self.subject, self.predicate, self.object)


# ── Prompt Template ───────────────────────────────────────────


CLAIM_EXTRACT_PROMPT = """You are a precise factual claim extractor. Your task is to extract atomic factual claims from the given text.

## Rules
1. Each claim must be an atomic, verifiable assertion about a specific entity.
2. Claims should be finer-grained than general relationships — capture specific attributes, quantities, dates, and conditions.
3. Assign a status to each claim:
   - "supporting": The text explicitly states or strongly implies this is true.
   - "contradicting": The text explicitly contradicts this.
   - "not_enough_info": Mentioned but without enough context to verify.
4. Include the exact evidence snippet (source_text) from the original text.
5. Confidence reflects extraction certainty (0.0-1.0), NOT truth of the claim itself.
6. Output ONLY valid JSON array.

## Text (Chunk {chunk_id}):
```
{text}
```

## Already Extracted Entities (for reference):
{entities_ctx}

## Already Extracted Relations (for reference):
{relations_ctx}

## Output Format
```json
[
  {{
    "subject": "entity_name",
    "predicate": "attribute_or_relation",
    "object": "value_or_target_entity",
    "description": "Natural language description of the claim",
    "status": "supporting|contradicting|not_enough_info",
    "confidence": 0.95,
    "source_text": "Exact quote from text supporting this claim",
    "start_char": <offset>,
    "end_char": <offset>
  }}
]
```

If no claims can be found, output an empty array: []"""


# ── Main Operator ─────────────────────────────────────────────


class ClaimExtract:
    """Extract atomic factual claims from text chunks.

    Integrates into the GraphRAG build pipeline after entity/relation
    extraction. Outputs structured claims that feed into:

    - Community Report generation (claims enrich community summaries)
    - Global Search (claims provide evidence for map-reduce reasoning)
    - Conflict detection (cross-document contradiction discovery)

    Usage:
        extractor = ClaimExtract(llm=extract_llm)
        context = extractor.run(context)
        # context["claims"] = [Claim(...), ...]
    """

    MAX_CLAIMS_PER_CHUNK = 15  # Cap to avoid hallucination
    MIN_CONFIDENCE = 0.3       # Filter low-confidence extractions
    BATCH_SIZE = 5             # Parallel LLM calls per batch

    def __init__(self, llm: BaseLLM = None):
        self._llm = llm

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract claims from all chunks in context.

        Reads from context:
            chunks: List of text chunk dicts.
            vertices: List of extracted entity dicts (optional, for reference).
            edges: List of extracted relation dicts (optional, for reference).
            doc_id: Source document ID (optional).

        Writes to context:
            claims: List of Claim objects.
            claim_count: Total number of extracted claims.
        """
        chunks = context.get("chunks", [])
        vertices = context.get("vertices", [])
        edges = context.get("edges", [])
        doc_id = context.get("doc_id", "unknown")

        if not chunks:
            log.warning("No chunks found for claim extraction.")
            context["claims"] = []
            context["claim_count"] = 0
            return context

        # Build entity/relation context for the prompt
        entities_ctx = self._format_entities(vertices)
        relations_ctx = self._format_relations(edges)

        all_claims = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            chunk_id = chunk.get("chunk_id", f"chunk_{i}") if isinstance(chunk, dict) else f"chunk_{i}"

            claims = self._extract_from_chunk(
                chunk_text=chunk_text,
                chunk_id=chunk_id,
                doc_id=doc_id,
                entities_ctx=entities_ctx,
                relations_ctx=relations_ctx,
            )
            all_claims.extend(claims)

        # Deduplicate by (subject, predicate, object)
        all_claims = self._deduplicate(all_claims)

        context["claims"] = [c.to_dict() for c in all_claims]
        context["claim_count"] = len(all_claims)
        log.info(
            "Claim extraction complete: %d claims from %d chunks",
            len(all_claims), len(chunks),
        )
        return context

    def _extract_from_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        doc_id: str,
        entities_ctx: str,
        relations_ctx: str,
    ) -> List[Claim]:
        """Extract claims from a single text chunk."""
        if not chunk_text or not chunk_text.strip():
            return []

        # Truncate very long chunks to save tokens
        truncated = chunk_text[:3000] if len(chunk_text) > 3000 else chunk_text

        prompt = CLAIM_EXTRACT_PROMPT.format(
            text=truncated,
            chunk_id=chunk_id,
            entities_ctx=entities_ctx or "(none provided)",
            relations_ctx=relations_ctx or "(none provided)",
        )

        try:
            if self._llm:
                response = self._llm.generate(prompt=prompt)
            else:
                response = "[]"
            claims = self._parse_response(response, chunk_id, doc_id)
        except Exception as e:
            log.error("Claim extraction failed for %s: %s", chunk_id, e)
            claims = []

        # Cap number of claims per chunk
        claims = claims[: self.MAX_CLAIMS_PER_CHUNK]

        # Filter by minimum confidence
        claims = [c for c in claims if c.confidence >= self.MIN_CONFIDENCE]

        return claims

    def _parse_response(self, response: str, chunk_id: str, doc_id: str) -> List[Claim]:
        """Parse LLM response into Claim objects."""
        # Try JSON extraction from markdown code fence
        json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try raw JSON array
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            json_str = json_match.group(0) if json_match else "[]"

        try:
            items = json.loads(json_str)
        except json.JSONDecodeError:
            log.warning("Failed to parse claim JSON: %s", response[:200])
            return []

        claims = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                # Validate required fields
                subject = item.get("subject", "").strip()
                predicate = item.get("predicate", "").strip()
                obj = item.get("object", "").strip()
                if not subject or not predicate or not obj:
                    continue  # Skip incomplete items

                # Map status string to enum
                raw_status = item.get("status", "not_enough_info")
                try:
                    status = ClaimStatus(raw_status)
                except ValueError:
                    status = ClaimStatus.NOT_ENOUGH_INFO

                claim = Claim(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    description=item.get("description", ""),
                    status=status,
                    confidence=float(item.get("confidence", 0.5)),
                    source_text=item.get("source_text", ""),
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    start_char=int(item.get("start_char", 0)),
                    end_char=int(item.get("end_char", 0)),
                )
                claims.append(claim)
            except (ValueError, KeyError, TypeError) as e:
                log.debug("Skipping invalid claim item: %s (%s)", item, e)
                continue

        return claims

    @staticmethod
    def _format_entities(entities: List[Dict]) -> str:
        """Format entity list for prompt context."""
        if not entities:
            return ""
        lines = []
        for v in entities[:30]:  # Limit to avoid oversized prompt
            label = v.get("label", "?")
            props = v.get("properties", {})
            name = props.get("name", v.get("id", "?"))
            lines.append(f"- [{label}] {name}")
        return "\n".join(lines)

    @staticmethod
    def _format_relations(relations: List[Dict]) -> str:
        """Format relation list for prompt context."""
        if not relations:
            return ""
        lines = []
        for e in relations[:30]:
            label = e.get("label", "")
            src = e.get("outV", "")
            dst = e.get("inV", "")
            lines.append(f"- ({src})-[{label}]->({dst})")
        return "\n".join(lines)

    @staticmethod
    def _deduplicate(claims: List[Claim]) -> List[Claim]:
        """Deduplicate claims by (subject, predicate, object) keeping highest confidence."""
        seen = {}
        for c in claims:
            key = (c.subject.lower().strip(), c.predicate.lower().strip(), c.object.lower().strip())
            existing = seen.get(key)
            if existing is None or c.confidence > existing.confidence:
                seen[key] = c
        return list(seen.values())


# ── Claim Index (in-memory for search) ────────────────────────


class ClaimIndex:
    """In-memory index for fast claim lookup and retrieval.

    Supports:
    - Subject-based lookup: find all claims about an entity.
    - Predicate-based lookup: find all claims of a given type.
    - Status filtering: find supporting / contradicting claims.
    - Community assignment: assign claims to communities based on subject.
    """

    def __init__(self):
        self._by_subject: Dict[str, List[Claim]] = {}
        self._by_predicate: Dict[str, List[Claim]] = {}
        self._by_status: Dict[str, List[Claim]] = {}
        self._all_claims: List[Claim] = []

    def add(self, claim: Claim) -> None:
        self._all_claims.append(claim)
        key_s = claim.subject.lower()
        key_p = claim.predicate.lower()
        key_v = claim.status.value if isinstance(claim.status, ClaimStatus) else claim.status
        self._by_subject.setdefault(key_s, []).append(claim)
        self._by_predicate.setdefault(key_p, []).append(claim)
        self._by_status.setdefault(key_v, []).append(claim)

    def add_batch(self, claims: List[Claim]) -> None:
        for c in claims:
            self.add(c)

    def get_by_subject(self, subject: str) -> List[Claim]:
        return self._by_subject.get(subject.lower(), [])

    def get_by_predicate(self, predicate: str) -> List[Claim]:
        return self._by_predicate.get(predicate.lower(), [])

    def get_by_status(self, status: str) -> List[Claim]:
        return self._by_status.get(status, [])

    def get_for_community(self, entity_ids: List[str]) -> List[Claim]:
        """Get all claims whose subject is in the given entity set (community)."""
        entity_set = {eid.lower() for eid in entity_ids}
        results = []
        for claim in self._all_claims:
            if claim.subject.lower() in entity_set:
                results.append(claim)
        return results

    @property
    def size(self) -> int:
        return len(self._all_claims)

    def stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        return {
            "total_claims": len(self._all_claims),
            "unique_subjects": len(self._by_subject),
            "unique_predicates": len(self._by_predicate),
            "status_breakdown": {
                k: len(v) for k, v in self._by_status.items()
            },
        }
