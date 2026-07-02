# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import asyncio
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from hugegraph_llm.config import huge_settings
from hugegraph_llm.indices.vector_index.base import VectorStoreBase
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManager
from hugegraph_llm.utils.log import log

# Valid strategy names
VALID_STRATEGIES = {"fastrag", "lightrag", "ms_graphrag", "hipporag"}


class BuildSemanticIndex:
    """Build and maintain vector index for graph vertex IDs.

    Supports 4 entity→vector text construction strategies (vid_embed_strategy):
      - fastrag (default): [{TYPE}] name\\n[DESCRIPTION] {properties} — type-aware rich semantics
      - lightrag: {name}\\n{description} — simple name+desc concat
      - ms_graphrag: {title}:{description} — colon-separated, dual-vector ready
      - hipporag: {name} only — pure name string (legacy HugeGraph behavior)
    """

    def __init__(self, embedding: BaseEmbedding, vector_index: type[VectorStoreBase]):
        self.vid_index = vector_index.from_name(embedding.get_embedding_dim(), huge_settings.graph_name, "graph_vids")
        self.embedding = embedding
        self.sm = SchemaManager(huge_settings.graph_name)

        # Resolve strategy from config
        self.strategy = getattr(huge_settings, "vid_embed_strategy", "fastrag").strip().lower()
        if self.strategy not in VALID_STRATEGIES:
            log.warning("Unknown vid_embed_strategy '%s', falling back to 'fastrag'. Valid: %s",
                        self.strategy, VALID_STRATEGIES)
            self.strategy = "fastrag"

        # Token/char budget for embed text
        self.max_chars = getattr(huge_settings, "vid_embed_max_chars", 512)

        # Properties to exclude from description text
        exclude_str = getattr(huge_settings, "vid_embed_exclude_props", "") or ""
        self.exclude_props = set(p.strip() for p in exclude_str.split(",") if p.strip())

        log.info("BuildSemanticIndex initialized: strategy=%s, max_chars=%d, exclude_props=%s",
                  self.strategy, self.max_chars, self.exclude_props or "(none)")

    # ──────────────────────────────────────────────
    # Strategy formatters: Dict[vid_or_detail] → str
    # ──────────────────────────────────────────────

    @staticmethod
    def _format_hipporag(vid: str, name: str) -> str:
        """HippoRAG2 style: raw name only."""
        return name

    @staticmethod
    def _format_lightrag(name: str, description: str) -> str:
        """LightRAG style: name + newline + description."""
        if description:
            return f"{name}\n{description}"
        return name

    @staticmethod
    def _format_ms_graphrag(name: str, description: str) -> str:
        """MS-GraphRAG style: title:description format."""
        if description:
            return f"{name}:{description}"
        return name

    @staticmethod
    def _format_fastrag(label: str, name: str, description: str) -> str:
        """Fast-GraphRAG style: [TYPE] name \\n [DESCRIPTION] description."""
        parts = [f"[{label.upper()}] {name}"]
        if description:
            parts.append(f"[DESCRIPTION] {description}")
        return "\n".join(parts)

    # ──────────────────────────────────────────────
    # Property → description text helpers
    # ──────────────────────────────────────────────

    def _properties_to_description(self, properties: Dict[str, Any]) -> str:
        """Convert vertex properties dict to a flat description string.

        Filters out excluded keys (IDs, timestamps), then joins as "key: value" pairs.
        """
        if not properties:
            return ""

        filtered = {}
        for k, v in properties.items():
            k_lower = k.strip().lower()
            if k_lower in self.exclude_props:
                continue
            # Skip None/empty values
            if v is None or v == "":
                continue
            filtered[k] = v

        if not filtered:
            return ""

        parts = [f"{k}: {v}" for k, v in sorted(filtered.items())]
        desc = ", ".join(parts)
        return self._truncate(desc)

    def _truncate(self, text: str) -> str:
        """Respect max_chars budget for embedding input."""
        if len(text) <= self.max_chars:
            return text
        truncated = text[:self.max_chars]
        # Avoid cutting mid-word
        last_space = truncated.rfind(" ")
        if last_space > self.max_chars * 0.8:
            truncated = truncated[:last_space]
        return truncated

    # ──────────────────────────────────────────────
    # Main dispatch: select formatter by strategy
    # ──────────────────────────────────────────────

    def _format_entity_text(self, vid_detail: Optional[Dict[str, Any]], vid: str, name: str) -> str:
        """Dispatch to the correct strategy formatter.

        Args:
            vid_detail: Enriched data from fetch_graph_data (may be None if legacy path).
                       Expected keys: {"label": str, "properties": dict}
            vid: Full vertex ID string, e.g. "Person:张三"
            name: Extracted display name (PK value or VID suffix)

        Returns:
            Text string to be embedded.
        """
        # Extract enriched fields (may be empty if not available)
        label = ""
        properties = {}
        if isinstance(vid_detail, dict):
            label = vid_detail.get("label", "")
            properties = vid_detail.get("properties") or {}

        # Build description from properties (shared by most strategies except hipporag)
        description = self._properties_to_description(properties)

        # Dispatch
        if self.strategy == "hipporag":
            return self._format_hipporag(vid, name)
        elif self.strategy == "lightrag":
            return self._format_lightrag(name, description)
        elif self.strategy == "ms_graphrag":
            return self._format_ms_graphrag(name, description)
        else:  # fastrag (default)
            effective_label = label or self._guess_label_from_vid(vid)
            return self._format_fastrag(effective_label, name, description)

    @staticmethod
    def _guess_label_from_vid(vid: str) -> str:
        """Fallback: extract label from VID format 'label_id:name'."""
        if not vid or ":" not in vid:
            return "ENTITY"
        raw = vid.split(":", 1)[0]
        # Strip trailing numeric suffix (e.g. "my_label_123" → "my_label")
        import re
        stripped = re.sub(r'_\d+$', '', raw)
        return stripped if stripped else "ENTITY"

    # ──────────────────────────────────────────────
    # Legacy backward-compat helpers
    # ──────────────────────────────────────────────

    def _extract_names(self, vertices: List[str]) -> List[str]:
        """Extract display names from VID strings (legacy helper).

        Takes the last segment after ':' as the display name.
        Handles both 2-part ("Person:张三") and 3-part ("Person:1:张三") VID formats.
        """
        return [v.rsplit(":", 1)[-1] if ":" in v else v for v in vertices]

    def _build_detail_lookup(self, vertex_details: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Build vid → detail lookup dict from enriched fetch result."""
        lookup: Dict[str, Dict[str, Any]] = {}
        for d in vertex_details:
            lookup[d["vid"]] = d
        return lookup

    # ──────────────────────────────────────────────
    # Embedding parallel execution
    # ──────────────────────────────────────────────

    async def _get_embeddings_parallel(self, texts: List[str]) -> List[Any]:
        sem = asyncio.Semaphore(10)
        batch_size = 1000

        async def get_embeddings_with_semaphore(text_list: List[str]) -> Any:
            async with sem:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self.embedding.get_texts_embeddings, text_list)

        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        tasks = [get_embeddings_with_semaphore(batch) for batch in batches]

        embeddings: List[Any] = []
        with tqdm(total=len(tasks), desc="Embedding batches") as pbar:
            for future in asyncio.as_completed(tasks):
                batch_embeddings = await future
                embeddings.extend(batch_embeddings)
                pbar.update(1)
        return embeddings

    # ──────────────────────────────────────────────
    # Main entry point (called by pipeline)
    # ──────────────────────────────────────────────

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        vertexlabels = self.sm.schema.getSchema()["vertexlabels"]
        all_pk_flag = bool(vertexlabels) and all(data.get("id_strategy") == "PRIMARY_KEY" for data in vertexlabels)

        past_vids = self.vid_index.get_all_properties()

        present_vids: List[str] = context.get("vertices", [])
        # Enriched details (new field from FetchGraphData._fetch_all_vertices_detail)
        vertex_details: List[Dict[str, Any]] = context.get("vertex_details", [])
        detail_lookup = self._build_detail_lookup(vertex_details) if vertex_details else {}

        removed_vids = set(past_vids) - set(present_vids)
        removed_num = self.vid_index.remove(removed_vids)
        added_vids = list(set(present_vids) - set(past_vids))

        if added_vids:
            # Build embed texts using configured strategy
            names = self._extract_names(added_vids) if all_pk_flag else list(added_vids)
            embed_texts: List[str] = []
            for idx, vid in enumerate(added_vids):
                name = names[idx] if idx < len(names) else vid
                vid_detail = detail_lookup.get(vid)
                text = self._format_entity_text(vid_detail, vid, name)
                embed_texts.append(text)

            added_embeddings = asyncio.run(self._get_embeddings_parallel(embed_texts))
            log.info(
                "Building vector index for %s vertices (strategy=%s)...",
                len(added_vids), self.strategy,
            )
            self.vid_index.add(added_embeddings, added_vids)
            self.vid_index.save_index_by_name(huge_settings.graph_name, "graph_vids")
        else:
            log.debug("No update vertices to build vector index.")

        context.update(
            {
                "removed_vid_vector_num": removed_num,
                "added_vid_vector_num": len(added_vids),
                "vid_embed_strategy": self.strategy,
            }
        )
        return context
