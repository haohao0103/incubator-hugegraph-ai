# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the vector index over the semantic layer.

Closes the ``vector_index=no`` capability gap flagged in M2/M3: without it
the MCP context tool sits at its weakest recall level and the score
thresholds (``min_seed_ratio`` / ``min_score_ratio``) are no-ops, because
rank-based RRF compresses scores into a band too narrow to discriminate.

Two pieces:

* :class:`SemanticIndexer` turns the projection into documents (tables and
  business terms), embeds them in batches and upserts into any
  :class:`~hugegraph_llm.nl2sql.vector_store.SchemaVectorStore`. Document ids
  use the ``table:`` / ``term:`` prefixes that
  :meth:`SemanticLayerRetriever._node_to_table` already resolves, so recall
  and expansion work without a mapping layer.
* :func:`embed_from_settings` adapts the project's configured embedding
  (``get_embedding`` -> ``get_text_embedding``) to the single-text callable
  the retriever expects. Failures surface at call time: this environment's
  configured endpoint returned 401, so the code path is tested with
  stand-ins and the real endpoint is a configuration fix, not a code one.

Batching matters more than it looks: a 33-table model is one small request,
but a 500-table warehouse is thousands, and per-text calls would make
indexing take minutes instead of seconds.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from hugegraph_llm.semantic_layer.readers import SemanticGraphReader
from hugegraph_llm.utils.log import log

__all__ = [
    "SemanticIndexer",
    "IndexStats",
    "embed_from_settings",
    "DeterministicEmbedder",
    "TABLE_PREFIX",
    "TERM_PREFIX",
]

#: Document id prefixes. Must stay aligned with
#: ``SemanticLayerRetriever._node_to_table``.
TABLE_PREFIX = "table:"
TERM_PREFIX = "term:"

EmbedFn = Callable[[str], List[float]]


@dataclass
class IndexStats:
    """What one indexing run produced."""

    tables: int = 0
    terms: int = 0
    embedded: int = 0
    failed: int = 0
    dimension: int = 0

    @property
    def documents(self) -> int:
        return self.tables + self.terms

    def to_dict(self) -> Dict[str, int]:
        return {
            "tables": self.tables,
            "terms": self.terms,
            "documents": self.documents,
            "embedded": self.embedded,
            "failed": self.failed,
            "dimension": self.dimension,
        }


def table_document(name: str, comment: str, columns: List[str], terms: List[str]) -> str:
    """Text representation of a table for embedding.

    Mirrors ``SemanticLayerRetriever._corpus``: column names are folded in so
    a query naming a column recalls the table that owns it. Business terms
    are included for the same reason -- "ARR" should reach ``subscriptions``
    even when no column carries that string.
    """
    parts = [name, comment, *columns, *terms]
    return " ".join(p for p in parts if p)


def term_document(name: str, description: str, aliases: List[str]) -> str:
    """Text representation of a business term."""
    parts = [name, description, *aliases]
    return " ".join(p for p in parts if p)


class SemanticIndexer:
    """Embeds the semantic layer into a vector store.

    :param retriever: supplies the projection and the corpus view; the same
        instance is later used for retrieval so document content and recall
        logic stay consistent by construction.
    :param vector_store: any ``SchemaVectorStore`` (numpy in-process, Milvus,
        OceanBase).
    :param embed: single-text embedding callable.
    """

    def __init__(
        self,
        retriever: Any,
        vector_store: Any,
        embed: EmbedFn,
        *,
        batch_size: int = 64,
    ) -> None:
        self.retriever = retriever
        self.vector_store = vector_store
        self.embed = embed
        self.batch_size = batch_size

    def build(self, refresh: bool = False) -> IndexStats:
        """Index every table and term in the projection.

        Re-embedding the whole catalogue is the default: metadata documents
        change as a whole (a new column changes its table's document), and
        per-document diffing adds state for little gain at catalogue scale.
        """
        if refresh:
            self.retriever.invalidate()

        projection = self.retriever.reader.projection()
        stats = IndexStats()

        ids: List[str] = []
        docs: List[str] = []
        for name, table in projection.tables.items():
            columns = [c.name for c in projection.columns_of(name)]
            ids.append(f"{TABLE_PREFIX}{name}")
            docs.append(table_document(
                name, table.comment, columns,
                projection.table_terms.get(name, []),
            ))
            stats.tables += 1
        for name, term in projection.terms.items():
            ids.append(f"{TERM_PREFIX}{name}")
            docs.append(term_document(name, term.description, term.aliases))
            stats.terms += 1

        if not docs:
            log.warning("semantic index: nothing to index (empty projection)")
            return stats

        vectors: List[List[float]] = []
        for start in range(0, len(docs), self.batch_size):
            batch = docs[start:start + self.batch_size]
            try:
                vectors.extend(self.embed(text) for text in batch)
                stats.embedded += len(batch)
            except Exception as exc:  # noqa: BLE001 - indexing is best-effort
                stats.failed += len(batch)
                log.warning(
                    "semantic index: embedding batch failed (%s texts): %s",
                    len(batch), exc,
                )

        if vectors:
            stats.dimension = len(vectors[0])
            embedded_ids = ids[:len(vectors)]
            self.vector_store.upsert(embedded_ids, vectors)

        log.info(
            "semantic index: %s/%s documents embedded (dim=%s)",
            stats.embedded, stats.documents, stats.dimension,
        )
        return stats


def embed_from_settings() -> EmbedFn:
    """Adapt the project-configured embedding to a single-text callable.

    Uses ``Embeddings().get_embedding()``, honouring ``EMBEDDING_TYPE`` and
    the OpenAI-compatible settings in the environment. The returned callable
    raises on failure -- callers decide whether that is fatal or a degradation
    to lexical-only recall.
    """
    from hugegraph_llm.models.embeddings.init_embedding import Embeddings

    backend = Embeddings().get_embedding()
    getter = getattr(backend, "get_text_embedding", None)
    if getter is None:
        raise TypeError(
            f"embedding backend {type(backend).__name__} has no "
            "get_text_embedding(); cannot adapt to the retriever contract"
        )

    def embed(text: str) -> List[float]:
        vector = getter(text)
        return [float(x) for x in vector]

    return embed


class DeterministicEmbedder:
    """Hash-based embedder for tests.

    Produces stable vectors from token hashes. It has **no semantic
    content**: similar texts are not close together. It exists to exercise
    the plumbing (batching, upsert, search, threshold behaviour) without a
    network, and tests must not use it to assert retrieval *quality*.
    """

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def __call__(self, text: str) -> List[float]:
        import hashlib

        vector = [0.0] * self.dimension
        for token in str(text).lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            # Deterministic sign keeps different tokens distinguishable.
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = sum(v * v for v in vector) ** 0.5
        if norm:
            vector = [v / norm for v in vector]
        return vector
