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
FAISS delete-by-ID without rebuilding the entire index.

Current FaissMemoryIndex.delete_memory() rebuilds the entire index from scratch,
which is O(n) and slow for large corpora. This module provides an efficient
alternative using faiss.IDSelector or a maintained ID→position mapping.

Two strategies:
  1. IndexIDMap: Wrap the base index with faiss.IndexIDMap, which maps
     external IDs to internal positions and supports remove_ids().
  2. Tombstone: Mark deleted IDs in a set; skip them during search.
     Periodically compact by rebuilding without tombstones.

Strategy 1 is preferred for moderate-size indexes (<1M vectors).
Strategy 2 is simpler and works for any size.
"""

import time
import threading
from typing import Any, Dict, List, Optional, Set

import numpy as np

from hugegraph_llm.utils.log import log


class FaissDeletableIndex:
    """FAISS index wrapper that supports efficient delete-by-ID.

    Uses faiss.IndexIDMap to map external integer IDs to internal positions,
    allowing remove_ids() without rebuilding. String IDs are mapped to
    integer IDs via a hash table.

    Args:
        dim: Embedding dimension.
        model_name: Sentence-transformers model name for embedding.
    """

    _model = None  # Shared model cache

    def __init__(self, dim: int = 384, model_name: str = "all-MiniLM-L6-v2"):
        self.dim = dim
        self.model_name = model_name
        self._lock = threading.RLock()

        # Internal ID sequence counter (must be unique per vector)
        self._next_id = 0

        # String ID → integer ID mapping
        self._str_to_int: Dict[str, int] = {}

        # Integer ID → metadata mapping
        self._int_to_meta: Dict[int, Dict[str, Any]] = {}

        # Build IndexIDMap over IndexFlatIP
        import faiss
        base_index = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap(base_index)

        # Tombstone set for deleted IDs (pending compaction)
        self._tombstones: Set[int] = set()

        # Load embedding model
        self._load_model()

    def _load_model(self):
        """Load sentence-transformers model (shared across instances)."""
        if FaissDeletableIndex._model is None:
            from sentence_transformers import SentenceTransformer
            FaissDeletableIndex._model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> np.ndarray:
        """Get embedding vector for text."""
        emb = FaissDeletableIndex._model.encode(
            text, convert_to_numpy=True, show_progress_bar=False
        )
        return emb.astype(np.float32)

    def add_memory(
        self,
        memory_id: str,
        content: str,
        created_at: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a memory vector with its string ID.

        Args:
            memory_id: Unique string identifier for this memory.
            content: Text content to embed and index.
            created_at: Timestamp of creation.
            metadata: Additional metadata dict.

        Returns:
            The internal integer ID assigned.
        """
        with self._lock:
            vec = self.embed_text(content).reshape(1, -1).astype(np.float32)
            int_id = self._next_id
            self._next_id += 1

            self.index.add_with_ids(vec, np.array([int_id], dtype=np.int64))

            self._str_to_int[memory_id] = int_id
            self._int_to_meta[int_id] = {
                "memory_id": memory_id,
                "content": content,
                "created_at": created_at or time.time(),
                "metadata": metadata or {},
                "int_id": int_id,
            }

            log.debug("Added memory %s (int_id=%d) to deletable index", memory_id, int_id)
            return int_id

    def remove_by_id(self, memory_id: str) -> bool:
        """Remove a vector by its string ID without rebuilding.

        Uses faiss.IndexIDMap.remove_ids() which is O(1) for removal
        (marks the slot as empty, no rebuild needed).

        Args:
            memory_id: String ID of the memory to remove.

        Returns:
            True if found and removed, False if not found.
        """
        with self._lock:
            int_id = self._str_to_int.get(memory_id)
            if int_id is None:
                return False

            # Remove from FAISS IndexIDMap
            import faiss
            ids_np = np.array([int_id], dtype=np.int64)
            n_removed = self.index.remove_ids(
                faiss.IDSelectorBatch(ids_np.size, faiss.swig_ptr(ids_np))
            )

            # Clean up mappings
            self._tombstones.add(int_id)
            del self._str_to_int[memory_id]
            if int_id in self._int_to_meta:
                del self._int_to_meta[int_id]

            log.debug(
                "Removed memory %s (int_id=%d, faiss_removed=%d)",
                memory_id, int_id, n_removed,
            )
            return n_removed > 0

    def remove_by_ids(self, memory_ids: List[str]) -> int:
        """Remove multiple vectors by their string IDs.

        Args:
            memory_ids: List of string IDs to remove.

        Returns:
            Number of successfully removed entries.
        """
        with self._lock:
            int_ids = []
            for mid in memory_ids:
                int_id = self._str_to_int.get(mid)
                if int_id is not None:
                    int_ids.append(int_id)

            if not int_ids:
                return 0

            import faiss
            ids_np = np.array(int_ids, dtype=np.int64)
            id_selector = faiss.IDSelectorBatch(ids_np.size, faiss.swig_ptr(ids_np))
            n_removed = self.index.remove_ids(id_selector)

            # Clean up mappings
            for mid in memory_ids:
                int_id = self._str_to_int.get(mid)
                if int_id is not None:
                    self._tombstones.add(int_id)
                    del self._str_to_int[mid]
                    if int_id in self._int_to_meta:
                        del self._int_to_meta[int_id]

            return n_removed

    def search(
        self,
        query: str,
        top_k: int = 5,
        ebbinghaus_weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Search memories by semantic similarity.

        Args:
            query: Search query text.
            top_k: Number of results.
            ebbinghaus_weights: Optional {memory_id: retention_score} weighting.

        Returns:
            List of result dicts with memory_id, content, score, retention.
        """
        with self._lock:
            if self.index.ntotal == 0:
                return []

            qvec = self.embed_text(query).reshape(1, -1).astype(np.float32)
            k = min(top_k * 3, self.index.ntotal)
            scores, indices = self.index.search(qvec, k)

            results = []
            seen: Set[str] = set()
            for score, idx in zip(scores[0], indices[0]):
                # IndexIDMap returns internal IDs, not positions
                int_id = int(idx)
                if int_id < 0 or int_id in self._tombstones:
                    continue

                meta = self._int_to_meta.get(int_id)
                if not meta:
                    continue

                mid = meta["memory_id"]
                if mid in seen:
                    continue
                seen.add(mid)

                raw_score = float(score)
                retention = 1.0
                if ebbinghaus_weights and mid in ebbinghaus_weights:
                    retention = ebbinghaus_weights[mid]
                weighted_score = raw_score * (0.3 + 0.7 * retention)

                results.append({
                    "memory_id": mid,
                    "content": meta["content"],
                    "score": weighted_score,
                    "raw_score": raw_score,
                    "retention": retention,
                    "int_id": int_id,
                })

            # Sort by weighted score descending
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        with self._lock:
            return {
                "total_vectors": self.index.ntotal,
                "total_known_ids": len(self._str_to_int),
                "tombstones": len(self._tombstones),
                "dim": self.dim,
            }

    def compact(self) -> int:
        """Rebuild the index without tombstones for space efficiency.

        This is equivalent to the old "rebuild" approach but only called
        periodically (e.g. every 1000 deletions) rather than on every delete.

        Returns:
            Number of vectors in the compacted index.
        """
        with self._lock:
            import faiss

            # Collect all live entries
            live_entries = []
            for int_id, meta in self._int_to_meta.items():
                if int_id not in self._tombstones:
                    live_entries.append((int_id, meta))

            # Rebuild from scratch
            base_index = faiss.IndexFlatIP(self.dim)
            new_index = faiss.IndexIDMap(base_index)
            self._next_id = 0
            self._str_to_int.clear()
            self._int_to_meta.clear()
            self._tombstones.clear()

            for old_int_id, meta in live_entries:
                vec = self.embed_text(meta["content"]).reshape(1, -1).astype(np.float32)
                new_int_id = self._next_id
                self._next_id += 1
                new_index.add_with_ids(vec, np.array([new_int_id], dtype=np.int64))
                self._str_to_int[meta["memory_id"]] = new_int_id
                self._int_to_meta[new_int_id] = {
                    **meta,
                    "int_id": new_int_id,
                }

            self.index = new_index
            log.info("Compacted FAISS index: %d vectors", self.index.ntotal)
            return self.index.ntotal
