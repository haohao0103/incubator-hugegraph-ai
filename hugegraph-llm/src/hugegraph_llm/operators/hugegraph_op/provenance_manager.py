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

"""Text provenance manager for Document → Chunk → Entity traceability.

Implements Neo4j-style Lexical Graph pattern: every entity and relationship
in the knowledge graph can be traced back to the document chunk it was
extracted from, enabling citation and audit capabilities.

Data model:
    (Document) -[CONTAINS_CHUNK]-> (Chunk) -[EXTRACTED_FROM]<- (Entity)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pyhugegraph.client import PyHugeClient

from hugegraph_llm.config import huge_settings
from hugegraph_llm.utils.log import log


# ── Provenance Data Model ─────────────────────────────────────

# HugeGraph schema labels for provenance (created at init time)
PROVENANCE_VERTEX_LABELS = {
    "Document": {
        "properties": ["name", "source", "created_at"],
        "nullable_keys": ["source", "created_at"],
        "primary_keys": ["name"],
    },
    "Chunk": {
        "properties": ["text", "index"],
        "nullable_keys": [],
        "primary_keys": ["text"],
    },
}

PROVENANCE_EDGE_LABELS = {
    "CONTAINS_CHUNK": {
        "source_label": "Document",
        "target_label": "Chunk",
        "properties": [],
    },
    "EXTRACTED_FROM": {
        "source_label": "Chunk",
        "target_label": "vertex",  # Reuses existing vertex label
        "properties": ["extraction_type"],
    },
}

PROVENANCE_PROPERTY_KEYS = [
    {"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "source", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "created_at", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "text", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "index", "data_type": "INT", "cardinality": "SINGLE"},
    {"name": "extraction_type", "data_type": "TEXT", "cardinality": "SINGLE"},
]


@dataclass
class ProvenanceRecord:
    """A single provenance link from entity → document chunk."""

    entity_id: str
    chunk_id: str
    chunk_text: str = ""
    chunk_index: int = 0
    document_name: str = ""
    document_source: str = ""
    extraction_type: str = "entity"

    def to_citation(self, max_text_len: int = 200) -> str:
        """Format as a citation string."""
        text = self.chunk_text[:max_text_len]
        if len(self.chunk_text) > max_text_len:
            text += "..."
        source = self.document_source or self.document_name
        return f"[{source}] 第{self.chunk_index}段: \"{text}\""


class ProvenanceManager:
    """Manages provenance chain creation and querying in HugeGraph.

    Creates Document → Chunk → Entity provenance edges during KG
    construction, and provides query methods for citation and audit.

    Usage:
        pm = ProvenanceManager()
        pm.init_schema()  # One-time schema setup
        doc_id = pm.create_document("report.pdf", "/data/report.pdf")
        chunk_id = pm.create_chunk(doc_id, "Sarah is 30 years old.", 0)
        pm.link_entity_to_chunk("1:Sarah", chunk_id, "entity")
        # Later:
        records = pm.get_provenance("1:Sarah")
    """

    # Vertex ID prefix for provenance nodes
    DOC_PREFIX = "DOC:"
    CHUNK_PREFIX = "CHUNK:"

    def __init__(self, client: PyHugeClient = None):
        """Initialize the provenance manager.

        Args:
            client: HugeGraph client. If None, creates from config.
        """
        if client is None:
            self._client = PyHugeClient(
                url=huge_settings.graph_url,
                graph=huge_settings.graph_name,
                user=huge_settings.graph_user,
                pwd=huge_settings.graph_pwd,
                graphspace=huge_settings.graph_space,
            )
        else:
            self._client = client
        self._schema = self._client.schema()
        self._initialized = False

    # ── Schema Initialization ─────────────────────────────────

    def init_schema(self) -> None:
        """Create provenance vertex/edge labels in HugeGraph (idempotent)."""
        if self._initialized:
            return

        # Create property keys
        for pk in PROVENANCE_PROPERTY_KEYS:
            try:
                prop_key = self._schema.propertyKey(pk["name"])
                if pk["data_type"] == "TEXT":
                    prop_key.asText()
                elif pk["data_type"] == "INT":
                    prop_key.asInt()
                if pk["cardinality"] == "SINGLE":
                    prop_key.valueSingle()
                prop_key.ifNotExist().create()
            except Exception as e:
                log.debug("Property key %s: %s", pk["name"], e)

        # Create vertex labels
        for label_name, config in PROVENANCE_VERTEX_LABELS.items():
            try:
                self._schema.vertexLabel(label_name).useCustomizeStringId().properties(
                    *config["properties"]
                ).nullableKeys(*config["nullable_keys"]).ifNotExist().create()
            except Exception as e:
                log.debug("Vertex label %s: %s", label_name, e)

        # Create edge labels
        for label_name, config in PROVENANCE_EDGE_LABELS.items():
            try:
                builder = (
                    self._schema.edgeLabel(label_name)
                    .sourceLabel(config["source_label"])
                    .targetLabel(config["target_label"])
                    .properties(*config["properties"])
                    .nullableKeys(*config["properties"])
                    .ifNotExist()
                    .create()
                )
            except Exception as e:
                log.debug("Edge label %s: %s", label_name, e)

        self._initialized = True
        log.info("Provenance schema initialized.")

    # ── Document Management ──────────────────────────────────

    def create_document(
        self, name: str, source: str = "", created_at: str = ""
    ) -> str:
        """Create a Document vertex.

        Args:
            name: Document name (used as custom vertex ID with DOC: prefix).
            source: File path or URL of the document.
            created_at: ISO timestamp.

        Returns:
            Document vertex ID (e.g., "DOC:report.pdf").
        """
        self.init_schema()
        doc_id = f"{self.DOC_PREFIX}{name}"
        props = {"name": name, "source": source, "created_at": created_at}
        try:
            self._client.graph().addVertex("Document", props, id=doc_id)
            log.debug("Created document: %s", doc_id)
        except Exception as e:
            log.debug("Document %s may already exist: %s", doc_id, e)
        return doc_id

    def create_chunk(
        self, doc_id: str, text: str, index: int
    ) -> str:
        """Create a Chunk vertex and link to its Document.

        Args:
            doc_id: The parent document ID (from create_document).
            text: The chunk text.
            index: Zero-based chunk index within the document.

        Returns:
            Chunk vertex ID.
        """
        self.init_schema()

        # Use text hash as stable ID
        import hashlib

        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        chunk_id = f"{self.CHUNK_PREFIX}{text_hash}"

        props = {"text": text, "index": index}
        try:
            self._client.graph().addVertex("Chunk", props, id=chunk_id)
            # Link Document → Chunk
            self._client.graph().addEdge("CONTAINS_CHUNK", doc_id, chunk_id, {})
            log.debug("Created chunk: %s[%d]", chunk_id, index)
        except Exception as e:
            log.debug("Chunk %s may already exist: %s", chunk_id, e)

        return chunk_id

    # ── Entity Linking ───────────────────────────────────────

    def link_entity_to_chunk(
        self, entity_id: str, chunk_id: str, extraction_type: str = "entity"
    ) -> bool:
        """Link an extracted entity to its source chunk.

        Args:
            entity_id: The entity's vertex ID in HugeGraph.
            chunk_id: The chunk vertex ID.
            extraction_type: "entity" or "relationship".

        Returns:
            True if link was created successfully.
        """
        self.init_schema()
        props = {"extraction_type": extraction_type}
        try:
            # Edge direction: Chunk → Entity (EXTRACTED_FROM)
            self._client.graph().addEdge("EXTRACTED_FROM", chunk_id, entity_id, props)
            return True
        except Exception as e:
            log.warning(
                "Failed to link entity %s to chunk %s: %s", entity_id, chunk_id, e
            )
            return False

    def link_batch(
        self, entity_ids: List[str], chunk_id: str, extraction_type: str = "entity"
    ) -> int:
        """Link multiple entities to the same source chunk.

        Returns:
            Number of successful links.
        """
        count = 0
        for eid in entity_ids:
            if self.link_entity_to_chunk(eid, chunk_id, extraction_type):
                count += 1
        return count

    # ── Provenance Query ─────────────────────────────────────

    def get_provenance(self, entity_id: str) -> List[ProvenanceRecord]:
        """Get the complete provenance chain for an entity.

        Traverses: Entity ←[EXTRACTED_FROM]← Chunk ←[CONTAINS_CHUNK]← Document

        Args:
            entity_id: The entity's vertex ID.

        Returns:
            List of ProvenanceRecord objects (may be multiple if entity
            was extracted from multiple chunks).
        """
        self.init_schema()
        records = []

        gremlin = f"""\
g.V('{entity_id}')
.inE('EXTRACTED_FROM').outV().as('chunk')
.inE('CONTAINS_CHUNK').outV().as('doc')
.select('chunk', 'doc')
.by(project('id', 'text', 'index')
    .by(id()).by(values('text')).by(values('index')))
.by(project('name', 'source')
    .by(values('name')).by(values('source')))
.toList()
"""
        try:
            result = self._client.gremlin().exec(gremlin)
            data = result.get("data", []) if isinstance(result, dict) else []
        except Exception as e:
            log.warning("Provenance query failed for %s: %s", entity_id, e)
            return records

        for item in data:
            chunk = item.get("chunk", {})
            doc = item.get("doc", {})
            records.append(
                ProvenanceRecord(
                    entity_id=entity_id,
                    chunk_id=chunk.get("id", ""),
                    chunk_text=chunk.get("text", ""),
                    chunk_index=chunk.get("index", 0),
                    document_name=doc.get("name", ""),
                    document_source=doc.get("source", ""),
                )
            )

        return records

    def get_chunk_entities(self, chunk_id: str) -> List[str]:
        """Get all entity IDs extracted from a specific chunk.

        Args:
            chunk_id: The chunk vertex ID.

        Returns:
            List of entity vertex IDs.
        """
        self.init_schema()

        gremlin = f"g.V('{chunk_id}').outE('EXTRACTED_FROM').inV().id().toList()"
        try:
            result = self._client.gremlin().exec(gremlin)
            return result.get("data", []) if isinstance(result, dict) else []
        except Exception as e:
            log.warning("Chunk entity query failed for %s: %s", chunk_id, e)
            return []

    def get_provenance_for_answer(
        self, entity_ids: List[str], max_per_entity: int = 2
    ) -> Dict[str, List[ProvenanceRecord]]:
        """Get provenance for all entities mentioned in an answer.

        Args:
            entity_ids: List of entity vertex IDs to trace.
            max_per_entity: Maximum citations per entity.

        Returns:
            Dict mapping entity_id → list of up to max_per_entity records.
        """
        result = {}
        for eid in entity_ids:
            records = self.get_provenance(eid)
            if records:
                result[eid] = records[:max_per_entity]
        return result


def create_provenance_manager(client: PyHugeClient = None) -> ProvenanceManager:
    """Create a ready-to-use ProvenanceManager."""
    pm = ProvenanceManager(client=client)
    pm.init_schema()
    return pm
