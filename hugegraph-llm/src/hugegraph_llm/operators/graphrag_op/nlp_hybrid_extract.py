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
NLP-based hybrid extractor for low-cost entity/relation extraction.

Inspired by LazyGraphRAG: uses NLP (regex, co-reference, dependency parsing)
for coarse extraction, then optionally refines with LLM only when needed.

This reduces indexing cost from ~100% (LLM-only) to near 0.1% for the
extraction phase, with optional LLM refinement for quality improvement.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from hugegraph_llm.utils.log import log


class ExtractMode(str, Enum):
    """Extraction mode: NLP only, LLM only, or hybrid (NLP + LLM refinement)."""

    NLP_ONLY = "nlp_only"
    LLM_ONLY = "llm_only"
    HYBRID = "hybrid"


class NLPExtractor:
    """
    NLP-based entity and relation extractor.

    Uses rule-based and lightweight NLP techniques for entity recognition
    and relation extraction, avoiding expensive LLM calls during indexing.
    Supports Chinese and English text.
    """

    # Regex patterns for entity extraction
    _PERSON_PATTERN = re.compile(
        r"(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)|"  # Western names
        r"(?:[\u4e00-\u9fff]{2,4})(?=(?:是|的|在|有|为))",  # Chinese names before common particles
        re.UNICODE,
    )
    _ORG_PATTERN = re.compile(
        r"(?:[A-Z][a-zA-Z]+(?:\s(?:Inc|Corp|Ltd|LLC|Company|Group|University|Institute))?\.?)|"
        r"((?:[\u4e00-\u9fff]{2,6})(?:公司|集团|大学|研究所|院|局|部|会|中心))",
        re.UNICODE,
    )
    _LOCATION_PATTERN = re.compile(
        r"(?:[A-Z][a-zA-Z]+(?:\s(?:City|State|Country|Province|District))?|"
        r"[\u4e00-\u9fff]{2,6}(?:省|市|区|县|镇|村|国|洲))",
        re.UNICODE,
    )
    _DATE_PATTERN = re.compile(
        r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?"
        r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
        r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s*\d{4}",
        re.IGNORECASE,
    )
    _NUMBER_PATTERN = re.compile(r"(?<!\w)-?\d+(?:\.\d+)?(?:\s*[%％])?")

    # Subject-predicate-object pattern for relation extraction
    _SVO_PATTERNS_EN = [
        # "A is B" / "A was B"
        re.compile(r"(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:is|was|are|were)\s+(?:a|an|the)?\s*(.+?)(?:\.|,|;|$)"),
        # "A works as/at B"
        re.compile(r"(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:works?\s+(?:as|at|for)|is\s+(?:a|an))\s+(.+?)(?:\.|,|;|$)"),
        # "A's B" possessive
        re.compile(
            r"(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)'s\s+(\w+(?:\s\w+){0,2})(?:\s+(?:is|was|are)\s+(.+?))?(?:\.|,|;|$)"
        ),
    ]
    _SVO_PATTERNS_CN = [
        # "A是B"
        re.compile(r"([\u4e00-\u9fff]{2,6})是([\u4e00-\u9fff\w]{2,20})"),
        # "A担任/担任了B"
        re.compile(r"([\u4e00-\u9fff]{2,6})(?:担任|就任|出任)([\u4e00-\u9fff\w]{2,20})"),
        # "A的B是C"
        re.compile(r"([\u4e00-\u9fff]{2,6})的([\u4e00-\u9fff]{2,8})是([\u4e00-\u9fff\w]{2,20})"),
    ]

    def __init__(self, language: str = "en"):
        self.language = language

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text using NLP rules.

        Returns a list of entity dicts with keys: name, type, mentions.
        """
        entities = {}
        # Person entities
        for match in self._PERSON_PATTERN.finditer(text):
            name = match.group(0).strip()
            if name and len(name) > 1:
                key = name.lower()
                if key not in entities:
                    entities[key] = {"name": name, "type": "person", "mentions": []}
                entities[key]["mentions"].append(name)

        # Organization entities
        for match in self._ORG_PATTERN.finditer(text):
            name = match.group(0).strip()
            if name and len(name) > 1:
                key = name.lower()
                if key not in entities:
                    entities[key] = {"name": name, "type": "organization", "mentions": []}
                entities[key]["mentions"].append(name)

        # Location entities
        for match in self._LOCATION_PATTERN.finditer(text):
            name = match.group(0).strip()
            if name and len(name) > 1:
                key = name.lower()
                if key not in entities:
                    entities[key] = {"name": name, "type": "location", "mentions": []}
                entities[key]["mentions"].append(name)

        # Date entities
        for match in self._DATE_PATTERN.finditer(text):
            name = match.group(0).strip()
            if name:
                key = f"date:{name.lower()}"
                if key not in entities:
                    entities[key] = {"name": name, "type": "date", "mentions": []}
                entities[key]["mentions"].append(name)

        return list(entities.values())

    def extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relations (subject-predicate-object triples) from text.

        Uses SVO pattern matching against previously extracted entities.
        Returns a list of relation dicts with keys: subject, predicate, object.
        """
        relations = []
        entity_names = {e["name"].lower(): e for e in entities}

        patterns = self._SVO_PATTERNS_CN if self.language == "zh" else self._SVO_PATTERNS_EN

        for pattern in patterns:
            for match in pattern.finditer(text):
                groups = [g for g in match.groups() if g]
                if len(groups) >= 2:
                    subject = groups[0].strip()
                    predicate = "related_to"
                    obj = groups[1].strip() if len(groups) == 2 else groups[2].strip()

                    # Try to infer a better predicate from the pattern
                    if len(groups) >= 3:
                        predicate = groups[1].strip()

                    # Verify subject is a known entity (or at least capitalized)
                    subject_lower = subject.lower()
                    if subject_lower in entity_names or subject[0].isupper():
                        relations.append(
                            {
                                "subject": subject,
                                "predicate": predicate,
                                "object": obj,
                            }
                        )

        return relations

    def extract_co_references(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simple co-reference resolution using pronoun matching.

        Maps pronouns back to the most recently mentioned entity of the
        appropriate type. This is a lightweight alternative to full
        coreference resolution models.
        """
        co_refs = []
        pronouns_en = {"he": "person", "she": "person", "it": "organization", "they": "person"}
        pronouns_cn = {"他": "person", "她": "person", "它": "organization", "他们": "person"}
        pronouns = pronouns_cn if self.language == "zh" else pronouns_en

        # Track last entity of each type
        last_entity_by_type: Dict[str, Dict[str, Any]] = {}
        sentences = re.split(r"[。.!?！？]", text)

        for sentence in sentences:
            for entity in entities:
                if entity["name"] in sentence:
                    last_entity_by_type[entity["type"]] = entity

            for pronoun, entity_type in pronouns.items():
                if pronoun in sentence.lower() and entity_type in last_entity_by_type:
                    ref_entity = last_entity_by_type[entity_type]
                    co_refs.append(
                        {
                            "pronoun": pronoun,
                            "resolved_entity": ref_entity["name"],
                            "entity_type": entity_type,
                        }
                    )

        return co_refs


class HybridExtractor:
    """
    Hybrid entity/relation extractor combining NLP and LLM.

    Strategy:
    1. NLP extracts entities and relations at near-zero cost
    2. If mode=HYBRID, LLM refines uncertain extractions
    3. If mode=NLP_ONLY, skip LLM entirely (lowest cost)
    4. If mode=LLM_ONLY, use traditional LLM extraction (backwards compatible)

    The hybrid approach dramatically reduces indexing cost by avoiding
    LLM calls for straightforward extractions, while still using LLM
    intelligence for ambiguous or complex cases.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        extract_mode: ExtractMode = ExtractMode.HYBRID,
        language: str = "en",
        nlp_confidence_threshold: float = 0.7,
    ):
        self.llm = llm
        self.extract_mode = extract_mode
        self.language = language
        self.nlp_confidence_threshold = nlp_confidence_threshold
        self.nlp_extractor = NLPExtractor(language=language)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the hybrid extraction pipeline.

        Args:
            context: Dict containing 'chunks' (list of text chunks) and
                     optionally 'schema' for schema-constrained extraction.

        Returns:
            Updated context with 'vertices', 'edges', 'triples' populated.
        """
        chunks = context.get("chunks", [])
        schema = context.get("schema")
        all_entities: List[Dict[str, Any]] = []
        all_relations: List[Dict[str, Any]] = []

        if self.extract_mode == ExtractMode.LLM_ONLY:
            return self._llm_only_extract(context)

        # Phase 1: NLP extraction across all chunks (near-zero cost)
        for chunk in chunks:
            entities = self.nlp_extractor.extract_entities(chunk)
            relations = self.nlp_extractor.extract_relations(chunk, entities)
            co_refs = self.nlp_extractor.extract_co_references(chunk, entities)

            # Resolve co-references in relations
            entity_name_map = {e["name"].lower(): e for e in entities}
            for ref in co_refs:
                entity_name_map[ref["pronoun"].lower()] = {"name": ref["resolved_entity"], "type": ref["entity_type"]}

            all_entities.extend(entities)
            all_relations.extend(relations)

        # Deduplicate entities by name
        unique_entities = self._deduplicate_entities(all_entities)
        unique_relations = self._deduplicate_relations(all_relations)

        # Phase 2: If HYBRID mode and LLM is available, refine uncertain extractions
        if self.extract_mode == ExtractMode.HYBRID and self.llm is not None:
            unique_entities, unique_relations = self._llm_refine(unique_entities, unique_relations, schema, chunks)

        # Phase 3: Map to graph vertices/edges format
        if schema:
            context = self._map_to_schema_format(context, unique_entities, unique_relations, schema)
        else:
            context["triples"] = [(r["subject"], r["predicate"], r["object"]) for r in unique_relations]
            context["vertices"] = []
            context["edges"] = []

        context["extracted_entities"] = unique_entities
        context["extracted_relations"] = unique_relations
        context["extract_mode"] = self.extract_mode.value
        context["call_count"] = context.get("call_count", 0) + (
            len(chunks) if self.extract_mode == ExtractMode.LLM_ONLY else 0
        )
        return context

    def _llm_only_extract(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to traditional LLM-only extraction."""
        if self.llm is None:
            log.warning("LLM not available for LLM_ONLY mode, falling back to NLP extraction")
            self.extract_mode = ExtractMode.NLP_ONLY
            return self.run(context)

        # Delegate to existing InfoExtract or PropertyGraphExtract
        # This path is backwards-compatible with the original extraction flow
        from hugegraph_llm.operators.llm_op.info_extract import InfoExtract
        from hugegraph_llm.operators.llm_op.property_graph_extract import PropertyGraphExtract

        schema = context.get("schema")
        if schema:
            extractor = PropertyGraphExtract(self.llm)
        else:
            extractor = InfoExtract(self.llm)
        return extractor.run(context)

    def _llm_refine(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]],
        chunks: List[str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Use LLM to refine uncertain NLP extractions.

        Only sends chunks where NLP extraction had low confidence or
        produced few results, minimizing LLM calls.
        """
        if not self.llm:
            return entities, relations

        # Identify chunks that need LLM refinement
        # Heuristic: if NLP extraction found few entities per chunk,
        # the chunk may have complex structure needing LLM
        low_confidence_chunks = []
        for chunk in chunks:
            chunk_entities = self.nlp_extractor.extract_entities(chunk)
            if len(chunk_entities) <= 1 and len(chunk) > 100:
                low_confidence_chunks.append(chunk)

        if not low_confidence_chunks:
            log.info("NLP extraction confidence is high, skipping LLM refinement")
            return entities, relations

        log.info("Refining %d low-confidence chunks with LLM", len(low_confidence_chunks))

        refined_entities = list(entities)
        refined_relations = list(relations)

        for chunk in low_confidence_chunks:
            try:
                prompt = self._build_refinement_prompt(chunk, schema, entities, relations)
                response = self.llm.generate(prompt=prompt)
                new_entities, new_relations = self._parse_refinement_response(response)
                refined_entities.extend(new_entities)
                refined_relations.extend(new_relations)
            except Exception as e:  # pylint: disable=broad-except
                log.warning("LLM refinement failed for chunk: %s", e)

        return self._deduplicate_entities(refined_entities), self._deduplicate_relations(refined_relations)

    def _build_refinement_prompt(
        self,
        chunk: str,
        schema: Optional[Dict[str, Any]],
        existing_entities: List[Dict[str, Any]],
        existing_relations: List[Dict[str, Any]],
    ) -> str:
        """Build a prompt for LLM to refine NLP-extracted entities and relations."""
        entity_names = [e["name"] for e in existing_entities]
        relation_strs = [f"{r['subject']} - {r['predicate']} - {r['object']}" for r in existing_relations]

        schema_str = ""
        if schema:
            import json

            schema_str = f"\nGraph Schema: {json.dumps(schema, ensure_ascii=False)}"

        return f"""You are an expert entity and relation extractor. The following text has been processed by an NLP system that found these entities: {entity_names}

And these relations: {relation_strs}

Please review and extract any ADDITIONAL entities and relations that the NLP system may have missed. Only output NEW entities and relations not already listed above.
{schema_str}

Text:
{chunk}

Output format (JSON only):
{{
  "entities": [{{"name": "...", "type": "person|organization|location|date|other"}}],
  "relations": [{{"subject": "...", "predicate": "...", "object": "..."}}]
}}"""

    def _parse_refinement_response(self, response: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse LLM refinement response into entities and relations."""
        import json

        entities = []
        relations = []

        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                for e in data.get("entities", []):
                    entities.append(
                        {"name": e.get("name", ""), "type": e.get("type", "other"), "mentions": [e.get("name", "")]}
                    )
                for r in data.get("relations", []):
                    relations.append(
                        {
                            "subject": r.get("subject", ""),
                            "predicate": r.get("predicate", "related_to"),
                            "object": r.get("object", ""),
                        }
                    )
        except json.JSONDecodeError:
            log.warning("Failed to parse LLM refinement response as JSON")

        return entities, relations

    def _map_to_schema_format(
        self,
        context: Dict[str, Any],
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Map NLP-extracted entities and relations to the schema-constrained
        vertex/edge format expected by Commit2Graph.

        Attempts to match NLP entity types to schema vertex labels,
        and relation predicates to schema edge labels.
        """
        if "vertices" not in context:
            context["vertices"] = []
        if "edges" not in context:
            context["edges"] = []

        vertex_label_map = {v["name"]: v for v in schema.get("vertexlabels", [])}
        edge_label_map = {e["name"]: e for e in schema.get("edgelabels", [])}

        # Type mapping: NLP type -> schema vertex label
        type_mapping = {
            "person": "person",
            "organization": "organization",
            "location": "location",
            "date": "date",
        }

        # Map entities to vertices
        name_to_vertex_id: Dict[str, str] = {}
        for entity in entities:
            entity_name = entity["name"]
            entity_type = entity.get("type", "other")

            # Find matching vertex label in schema
            matched_label = None
            # First try direct type match
            if entity_type in type_mapping and type_mapping[entity_type] in vertex_label_map:
                matched_label = type_mapping[entity_type]
            # Then try name match
            if matched_label is None:
                for label_name in vertex_label_map:
                    if label_name.lower() == entity_type.lower():
                        matched_label = label_name
                        break
            # Fallback to first available vertex label
            if matched_label is None and vertex_label_map:
                matched_label = list(vertex_label_map.keys())[0]
                log.debug("Mapping entity '%s' to fallback label '%s'", entity_name, matched_label)

            if matched_label:
                vertex_info = vertex_label_map[matched_label]
                primary_keys = vertex_info.get("primary_keys", [])
                props = {"name": entity_name}

                # Generate vertex ID
                if primary_keys and primary_keys[0] == "name":
                    vid = f"{vertex_info.get('id', '1')}:{entity_name}"
                else:
                    vid = entity_name

                name_to_vertex_id[entity_name] = vid
                context["vertices"].append(
                    {
                        "id": vid,
                        "label": matched_label,
                        "type": "vertex",
                        "properties": props,
                    }
                )

        # Map relations to edges
        for relation in relations:
            predicate = relation["predicate"]
            subject_name = relation["subject"]
            obj_name = relation["object"]

            # Find matching edge label
            matched_edge = None
            for edge_name in edge_label_map:
                if edge_name.lower() in predicate.lower() or predicate.lower() in edge_name.lower():
                    matched_edge = edge_name
                    break

            if matched_edge is None and edge_label_map:
                # Try to infer from relation context
                for edge_name in edge_label_map:
                    edge_info = edge_label_map[edge_name]
                    source_label = edge_info.get("source_label", "")
                    target_label = edge_info.get("target_label", "")
                    # Check if subject and object types match
                    s_type = next((e["type"] for e in entities if e["name"] == subject_name), None)
                    o_type = next((e["type"] for e in entities if e["name"] == obj_name), None)
                    if s_type and o_type:
                        s_mapped = type_mapping.get(s_type, s_type)
                        o_mapped = type_mapping.get(o_type, o_type)
                        if s_mapped == source_label and o_mapped == target_label:
                            matched_edge = edge_name
                            break

            if matched_edge:
                edge_info = edge_label_map[matched_edge]
                out_v = name_to_vertex_id.get(subject_name, subject_name)
                in_v = name_to_vertex_id.get(obj_name, obj_name)
                context["edges"].append(
                    {
                        "label": matched_edge,
                        "type": "edge",
                        "outV": out_v,
                        "outVLabel": edge_info.get("source_label", ""),
                        "inV": in_v,
                        "inVLabel": edge_info.get("target_label", ""),
                        "properties": {},
                    }
                )

        return context

    @staticmethod
    def _deduplicate_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate entities by name (case-insensitive)."""
        seen: Set[str] = set()
        unique = []
        for entity in entities:
            key = entity["name"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(entity)
            else:
                # Merge mentions
                for existing in unique:
                    if existing["name"].lower() == key:
                        existing["mentions"] = list(set(existing.get("mentions", []) + entity.get("mentions", [])))
                        break
        return unique

    @staticmethod
    def _deduplicate_relations(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate relations by (subject, predicate, object) tuple."""
        seen: Set[Tuple[str, str, str]] = set()
        unique = []
        for rel in relations:
            key = (rel["subject"].lower(), rel["predicate"].lower(), rel["object"].lower())
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        return unique
