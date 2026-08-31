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

# pylint: disable=W0621

import json
import re
from typing import Any, Dict, List

from hugegraph_llm.config import prompt
from hugegraph_llm.document.chunk_split import ChunkSplitter
from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.utils.log import log

# TODO: It is not clear whether there is any other dependence on the SCHEMA_EXAMPLE_PROMPT variable.
# Because the SCHEMA_EXAMPLE_PROMPT variable will no longer change based on
# prompt.extract_graph_prompt changes after the system loads, this does not seem to meet expectations.
SCHEMA_EXAMPLE_PROMPT = prompt.extract_graph_prompt


# Gleaning (multi-round completion), mirroring microsoft/graphrag's
# CONTINUE_PROMPT / LOOP_PROMPT pair. The Y/N gate is what keeps this cheap:
# when the first pass was already complete we spend one short call instead of
# running a second full extraction.
GLEANING_CONTINUE_PROMPT = (
    "MANY vertices and edges were missed in the last extraction. "
    "Remember to ONLY emit vertices/edges that match the given schema. "
    "Add ONLY the ones that were missed, using the same JSON format:\n"
)
GLEANING_LOOP_PROMPT = (
    "It appears some vertices or edges may have still been missed. "
    "Answer Y if there are still vertices or edges that need to be added, "
    "or N if there are none. Answer with a single letter: Y or N.\n"
)


def generate_extract_property_graph_prompt(text, schema=None, extra_instruction: str = "") -> str:
    hint = f"\n{extra_instruction}\n" if extra_instruction else ""
    return f"""---
Following the full instructions above, try to extract the following text from the given schema, output the JSON result:
{hint}# Input
## Text:
{text}
## Graph schema
{schema}

# Output"""


def split_text(text: str) -> List[str]:
    chunk_splitter = ChunkSplitter(split_type="paragraph", language="zh")
    chunks = chunk_splitter.split(text)
    return chunks


def balance_curly_braces(json_string: str) -> str:
    """Append the missing closing brackets, ignoring any inside string values.

    A truncated LLM response usually just loses its trailing ``}`` (or ``]``).
    Repairing that is cheap; losing the whole chunk is not. Approach mirrors
    neo4j-graphrag's helper, extended to square brackets so truncated arrays
    are recovered too.
    """
    stack: List[str] = []
    fixed: List[str] = []
    in_string = False
    escape = False

    for char in json_string:
        if char == '"' and not escape:
            in_string = not in_string
        elif char == "\\" and in_string:
            escape = not escape
            fixed.append(char)
            continue
        else:
            escape = False

        if not in_string:
            if char in "{[":
                stack.append(char)
                fixed.append(char)
            elif char in "}]":
                opener = "{" if char == "}" else "["
                if stack and stack[-1] == opener:
                    stack.pop()
                    fixed.append(char)
                else:
                    continue  # unmatched closer — drop it
            else:
                fixed.append(char)
        else:
            fixed.append(char)

    while stack:
        fixed.append("}" if stack.pop() == "{" else "]")
    return "".join(fixed)


def _repair_json(raw_json: str) -> str:
    """Best-effort repair of the JSON defects LLMs actually produce.

    Deliberately dependency-free: neo4j-graphrag leans on the third-party
    ``json_repair`` package, which this module does not carry. Covers the two
    failures that dominate in practice — trailing commas and a tail cut off by
    the output token limit.
    """
    text = re.sub(r",\s*([}\]])", r"\1", raw_json)  # trailing commas
    return balance_curly_braces(text)               # truncated tail


def filter_item(schema, items) -> List[Dict[str, Any]]:
    # filter vertex and edge with invalid properties
    filtered_items = []
    properties_map = {"vertex": {}, "edge": {}}
    for vertex in schema["vertexlabels"]:
        properties_map["vertex"][vertex["name"]] = {
            "primary_keys": vertex["primary_keys"],
            "nullable_keys": vertex.get("nullable_keys", []),
            "properties": vertex["properties"],
        }
    for edge in schema["edgelabels"]:
        properties_map["edge"][edge["name"]] = {"properties": edge["properties"]}
    log.info("properties_map: %s", properties_map)
    for item in items:
        item_type = item["type"]
        if item_type in properties_map:
            label = item["label"]
            item["properties"] = {
                key: value
                for key, value in item["properties"].items()
                if key in properties_map[item_type][label]["properties"]
            }
        filtered_items.append(item)

    return filtered_items


class PropertyGraphExtract:
    # Maximum characters per LLM call to avoid gateway timeouts (e.g. nginx 60s).
    # ~500 chars ≈ 2 paragraphs, balancing timeout safety (~40-50s per call) with
    # enough cross-entity context for the LLM to identify relationships (edges).
    MAX_CHUNK_CHARS = 500

    def __init__(
        self,
        llm: BaseLLM,
        example_prompt: str = prompt.extract_graph_prompt,
        gleaning: bool = False,
        max_gleanings: int = 3,
    ) -> None:
        self.llm = llm
        self.example_prompt = example_prompt
        # Gleaning: after the first pass per sub-chunk, ask the LLM (Y/N) whether
        # anything was missed and re-extract with the CONTINUE hint if so.
        # Default OFF — behaviour is unchanged unless explicitly enabled.
        self.gleaning = gleaning
        self.max_gleanings = max_gleanings
        self.NECESSARY_ITEM_KEYS = {"label", "type", "properties"}  # pylint: disable=invalid-name

    @staticmethod
    def _split_into_subchunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
        """Split a large text chunk into smaller sub-chunks by paragraph boundaries."""
        paragraphs = re.split(r"\n{2,}", text.strip())
        sub_chunks: List[str] = []
        current = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if current and len(current) + len(para) + 2 > max_chars:
                sub_chunks.append(current)
                current = para
            else:
                current = f"{current}\n\n{para}".strip() if current else para
        if current:
            sub_chunks.append(current)
        return sub_chunks if sub_chunks else [text]

    def run(self, context: Dict[str, Any]) -> Dict[str, List[Any]]:
        schema = context["schema"]
        chunks = context["chunks"]
        if "vertices" not in context:
            context["vertices"] = []
        if "edges" not in context:
            context["edges"] = []

        all_parsed_vertices = []
        all_parsed_edges = []
        discarded_items = []
        total_ll_calls = 0

        for chunk in chunks:
            sub_chunks = self._split_into_subchunks(chunk) if len(chunk) > self.MAX_CHUNK_CHARS else [chunk]
            for sub_chunk in sub_chunks:
                proceeded_chunk = self.extract_property_graph_by_llm(schema, sub_chunk)
                total_ll_calls += 1
                log.debug(
                    "[LLM] %s input (sub-chunk): %s \n output:%s",
                    self.__class__.__name__,
                    sub_chunk[:200],
                    proceeded_chunk[:500] if proceeded_chunk else "",
                )
                parsed = self._parse_extracted_graph(schema, proceeded_chunk)
                all_parsed_vertices.extend(parsed["vertices"])
                all_parsed_edges.extend(parsed["edges"])
                discarded_items.extend(parsed.get("discarded", []))

                if self.gleaning:
                    # Only spend extra LLM calls when the model says something is
                    # still missing (Y/N gate), and bail out as soon as a round
                    # returns nothing new.
                    for _ in range(self.max_gleanings):
                        if not self._should_glean(schema, sub_chunk, all_parsed_vertices,
                                                  all_parsed_edges):
                            break
                        total_ll_calls += 1
                        more = self.extract_property_graph_by_llm(
                            schema, sub_chunk, extra_instruction=GLEANING_CONTINUE_PROMPT)
                        parsed_more = self._parse_extracted_graph(schema, more)
                        if not parsed_more["vertices"] and not parsed_more["edges"]:
                            break
                        all_parsed_vertices.extend(parsed_more["vertices"])
                        all_parsed_edges.extend(parsed_more["edges"])
                        discarded_items.extend(parsed_more.get("discarded", []))

        # Build schema maps
        vertex_label_map = {v["name"]: v for v in schema["vertexlabels"]}
        edge_label_map = {e["name"]: e for e in schema["edgelabels"]}

        # Normalize ALL vertices first → complete ID map
        vertices, vertex_id_map = self._normalize_vertices(all_parsed_vertices, vertex_label_map)
        # Add fuzzy matching entries for tolerant edge endpoint resolution
        self._add_fuzzy_vertex_ids(vertices, vertex_label_map, vertex_id_map)

        # Resolve edges using the COMPLETE vertex map (deferred from per-chunk resolution)
        edges = self._normalize_edges(all_parsed_edges, edge_label_map, vertex_label_map, vertex_id_map)

        # Apply property filtering
        all_items = vertices + edges
        all_items = filter_item(schema, all_items)

        for item in all_items:
            if item["type"] == "vertex":
                context["vertices"].append(item)
            elif item["type"] == "edge":
                context["edges"].append(item)

        if discarded_items:
            context.setdefault("discarded_items", [])
            context["discarded_items"].extend(discarded_items)

        context["call_count"] = context.get("call_count", 0) + total_ll_calls
        final_v = sum(1 for i in all_items if i["type"] == "vertex")
        final_e = sum(1 for i in all_items if i["type"] == "edge")
        log.info(
            "PropertyGraphExtract: %d LLM calls for %d chunks → %d vertices + %d edges "
            "(raw: %d vertices, %d edges before resolution)",
            total_ll_calls,
            len(chunks),
            final_v,
            final_e,
            len(all_parsed_vertices),
            len(all_parsed_edges),
        )
        return context

    def _parse_extracted_graph(self, schema, text) -> Dict[str, List[Dict[str, Any]]]:
        """Parse LLM output into raw vertex and edge dicts without endpoint resolution."""
        result: Dict[str, List[Dict[str, Any]]] = {"vertices": [], "edges": [], "discarded": []}

        text = re.sub(r"```\w*\n?", "", text)
        text = re.sub(r"```", "", text)
        text = text.strip()

        json_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if not json_match:
            log.critical("Invalid property graph! No JSON found, please check the output format example in prompt.")
            return result
        json_str = json_match.group(1).strip()

        try:
            try:
                property_graph = json.loads(json_str)
            except json.JSONDecodeError:
                # Best-effort repair before discarding the whole chunk.
                property_graph = json.loads(_repair_json(json_str))
                log.warning("Recovered malformed extraction JSON via repair pass.")
            if isinstance(property_graph, list):
                vertices = [i for i in property_graph if isinstance(i, dict) and i.get("type") == "vertex"]
                edges = [i for i in property_graph if isinstance(i, dict) and i.get("type") == "edge"]
                property_graph = {"vertices": vertices, "edges": edges}
            if not (isinstance(property_graph, dict) and "vertices" in property_graph and "edges" in property_graph):
                log.critical("Invalid property graph format; expecting 'vertices' and 'edges'.")
                return result

            vertex_label_set = {v["name"] for v in schema["vertexlabels"]}
            edge_label_set = {e["name"] for e in schema["edgelabels"]}

            for item_type, item_list, valid_labels in [
                ("vertex", property_graph["vertices"], vertex_label_set),
                ("edge", property_graph["edges"], edge_label_set),
            ]:
                for item in item_list:
                    if not isinstance(item, dict):
                        continue
                    item = dict(item)
                    item_type_value = item.get("type", item_type)
                    item["type"] = item_type_value
                    if not self.NECESSARY_ITEM_KEYS.issubset(item.keys()):
                        continue
                    if item_type_value != item_type:
                        continue
                    if item["label"] not in valid_labels:
                        log.warning("Invalid %s label '%s' ignored.", item_type, item["label"])
                        discarded = {
                            "type": item_type,
                            "label": item["label"],
                            "properties": item.get("properties", {}),
                        }
                        if item_type == "edge":
                            out_label = item.get("outVLabel", "")
                            in_label = item.get("inVLabel", "")
                            source = item.get("source")
                            target = item.get("target")
                            if isinstance(source, dict):
                                out_label = out_label or source.get("label", "")
                            if isinstance(target, dict):
                                in_label = in_label or target.get("label", "")
                            discarded["outVLabel"] = out_label
                            discarded["inVLabel"] = in_label
                        result["discarded"].append(discarded)
                        continue
                    key = "vertices" if item_type == "vertex" else "edges"
                    result[key].append(item)

        except json.JSONDecodeError:
            log.critical("Invalid property graph JSON! Please check the extracted JSON data carefully")
        return result

    @staticmethod
    def _add_fuzzy_vertex_ids(vertices, vertex_label_map, vertex_id_map):
        """Add extra entries to vertex_id_map for fuzzy edge endpoint matching.

        The LLM may generate vertex IDs in various formats (plain name, label:name, etc.)
        that don't match the canonical {labelId}:{primaryKey} format. This method adds
        lookup entries for common alternative formats so edges can still find their endpoints.
        """
        for vertex in vertices:
            label = vertex["label"]
            vid = vertex.get("id")
            props = vertex.get("properties", {})
            vl = vertex_label_map.get(label, {})
            for pk in vl.get("primary_keys", []):
                pk_val = props.get(pk)
                if pk_val:
                    # Allow matching by plain primary key value
                    vertex_id_map.setdefault((label, str(pk_val)), vid)
                    # Allow matching by label:value format (e.g. "company:摩拜单车")
                    vertex_id_map.setdefault((label, f"{label}:{pk_val}"), vid)
                    # Allow matching by label name:value (e.g. "company:摩拜单车" keyed by Chinese label)
                    vertex_id_map.setdefault((label, f"{label}:{pk_val}"), vid)

    def extract_property_graph_by_llm(self, schema, chunk, extra_instruction: str = ""):
        prompt = generate_extract_property_graph_prompt(chunk, schema, extra_instruction)
        if self.example_prompt is not None:
            prompt = self.example_prompt + prompt
        return self.llm.generate(prompt=prompt)

    def _should_glean(self, schema, chunk, vertices, edges) -> bool:
        """Ask the LLM whether the last pass still missed anything (Y/N gate).

        Mirrors microsoft/graphrag's LOOP_PROMPT. This is deliberately a tiny
        prompt so the "already complete" case costs one short call instead of a
        second full extraction. Any non-Y answer (or any failure) stops the loop.
        """
        already = {
            "vertices": [v.get("id") or v.get("label") for v in vertices[-50:]],
            "edges": [e.get("label") for e in edges[-50:]],
        }
        prompt = (
            f"{self.example_prompt or ''}\n"
            f"# Input\n## Text:\n{chunk}\n## Graph schema\n{schema}\n\n"
            f"## Already extracted (last {len(already['vertices'])} vertices / "
            f"{len(already['edges'])} edges)\n{already}\n\n"
            f"# Question\n{GLEANING_LOOP_PROMPT}"
        )
        try:
            answer = self.llm.generate(prompt=prompt)
        except Exception as exc:  # noqa: BLE001 -- never fail extraction on the gate
            log.warning("gleaning gate failed, stopping loop: %s", exc)
            return False
        return str(answer).strip().upper().startswith("Y")

    @staticmethod
    def _primary_key_id(vertex_label, properties):
        id_strategy = vertex_label.get("id_strategy")
        if id_strategy and str(id_strategy).upper() != "PRIMARY_KEY":
            return None
        primary_keys = vertex_label.get("primary_keys", [])
        if not primary_keys or "id" not in vertex_label:
            return None
        values = []
        for key in primary_keys:
            value = properties.get(key)
            if value is None or value == "":
                return None
            values.append(str(value))
        return f"{vertex_label['id']}:{'!'.join(values)}"

    def _normalize_vertices(self, vertices, vertex_label_map):
        vertex_id_map = {}
        normalized_vertices = []
        for vertex in vertices:
            label = vertex["label"]
            properties = vertex["properties"]
            canonical_id = self._primary_key_id(vertex_label_map[label], properties)
            original_id = vertex.get("id")
            if canonical_id is None:
                if original_id:
                    vertex_id_map[(label, original_id)] = original_id
                normalized_vertices.append(vertex)
                continue

            vertex["id"] = canonical_id
            vertex_id_map[(label, canonical_id)] = canonical_id
            if original_id:
                vertex_id_map[(label, original_id)] = canonical_id
            normalized_vertices.append(vertex)
        return normalized_vertices, vertex_id_map

    def _resolve_endpoint(self, edge, endpoint_key, label_key, legacy_key, vertex_label_map, vertex_id_map):
        endpoint = edge.get(endpoint_key)
        label = edge.get(label_key)
        if endpoint and label:
            return vertex_id_map.get((label, endpoint)), label

        legacy_endpoint = edge.get(legacy_key)
        if not isinstance(legacy_endpoint, dict):
            return None, label

        label = legacy_endpoint.get("label")
        properties = legacy_endpoint.get("properties", {})
        if label not in vertex_label_map:
            return None, label
        canonical_id = self._primary_key_id(vertex_label_map[label], properties)
        return vertex_id_map.get((label, canonical_id)), label

    def _normalize_edges(self, edges, edge_label_map, vertex_label_map, vertex_id_map):
        normalized_edges = []
        for edge in edges:
            edge_label = edge_label_map[edge["label"]]
            out_v, out_v_label = self._resolve_endpoint(
                edge,
                "outV",
                "outVLabel",
                "source",
                vertex_label_map,
                vertex_id_map,
            )
            in_v, in_v_label = self._resolve_endpoint(
                edge,
                "inV",
                "inVLabel",
                "target",
                vertex_label_map,
                vertex_id_map,
            )
            if not out_v or not in_v:
                log.warning("Invalid edge endpoints '%s' have been ignored.", edge)
                continue
            if out_v_label != edge_label.get("source_label") or in_v_label != edge_label.get("target_label"):
                log.warning("Invalid edge endpoint labels '%s' have been ignored.", edge)
                continue

            edge["outV"] = out_v
            edge["outVLabel"] = out_v_label
            edge["inV"] = in_v
            edge["inVLabel"] = in_v_label
            normalized_edges.append(edge)
        return normalized_edges

    def _extract_and_filter_label(self, schema, text) -> List[Dict[str, Any]]:
        # Strip markdown code blocks (e.g. ```json ... ```)
        text = re.sub(r"```\w*\n?", "", text)
        text = re.sub(r"```", "", text)
        text = text.strip()

        # Try to extract JSON (object or array)
        json_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if not json_match:
            log.critical("Invalid property graph! No JSON found, please check the output format example in prompt.")
            return []
        json_str = json_match.group(1).strip()

        items = []
        try:
            property_graph = json.loads(json_str)
            # Handle flat array format: convert to {"vertices": [...], "edges": [...]}
            if isinstance(property_graph, list):
                vertices = [item for item in property_graph if isinstance(item, dict) and item.get("type") == "vertex"]
                edges = [item for item in property_graph if isinstance(item, dict) and item.get("type") == "edge"]
                property_graph = {"vertices": vertices, "edges": edges}
            # Expect property_graph to be a dict with keys "vertices" and "edges"
            if not (isinstance(property_graph, dict) and "vertices" in property_graph and "edges" in property_graph):
                log.critical("Invalid property graph format; expecting 'vertices' and 'edges'.")
                return items

            # Create sets for valid vertex and edge labels based on the schema
            vertex_label_map = {vertex["name"]: vertex for vertex in schema["vertexlabels"]}
            edge_label_map = {edge["name"]: edge for edge in schema["edgelabels"]}
            vertex_label_set = set(vertex_label_map)
            edge_label_set = set(edge_label_map)

            def process_items(item_list, valid_labels, item_type):
                parsed_items = []
                for item in item_list:
                    if not isinstance(item, dict):
                        log.warning("Invalid property graph item type '%s'.", type(item))
                        continue
                    item = dict(item)
                    item_type_value = item.get("type", item_type)
                    item["type"] = item_type_value
                    if not self.NECESSARY_ITEM_KEYS.issubset(item.keys()):
                        log.warning("Invalid item keys '%s'.", item.keys())
                        continue
                    if item_type_value != item_type:
                        log.warning("Invalid %s type '%s' has been ignored.", item_type, item_type_value)
                        continue
                    if item["label"] not in valid_labels:
                        log.warning(
                            "Invalid %s label '%s' has been ignored.",
                            item_type,
                            item["label"],
                        )
                        continue
                    parsed_items.append(item)
                return parsed_items

            vertex_items = process_items(property_graph["vertices"], vertex_label_set, "vertex")
            vertices, vertex_id_map = self._normalize_vertices(vertex_items, vertex_label_map)
            edge_items = process_items(property_graph["edges"], edge_label_set, "edge")
            edges = self._normalize_edges(edge_items, edge_label_map, vertex_label_map, vertex_id_map)
            items = vertices + edges
        except json.JSONDecodeError:
            log.critical("Invalid property graph JSON! Please check the extracted JSON data carefully")
        return items
