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
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Memory compression and clustering (PowerMem MemoryOptimizer aligned).

PowerMem's MemoryOptimizer.compress() performs:
  1. Semantic clustering of similar memories
  2. LLM summarization of each cluster
  3. Archiving original memories, replacing with summaries
  4. Importance-weighted pruning of low-value memories

We implement:
  - MemoryCompressor: clustering + LLM summarization + archival
  - ClusterFinder: k-means or affinity-based clustering of memory vectors
  - SummaryGenerator: LLM-based cluster summarization
  - PruningEngine: importance/Ebbinghaus-based memory pruning
"""

import hashlib
import json
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from hugegraph_llm.config.memory_config import memory_settings
from hugegraph_llm.engines.memory.intelligence import EbbinghausDecay, ImportanceEvaluator
from hugegraph_llm.utils.log import log


# LLM prompt for cluster summarization
CLUSTER_SUMMARY_PROMPT = """Summarize the following group of related memories into a single concise summary.
Keep all important facts, names, dates, and relationships. Remove redundancies.

Memories:
{memories}

Output a single summary paragraph, no more than 3 sentences."""


class ClusterFinder:
    """Find clusters of similar memories using embedding vectors.

    Uses simple k-means or centroid-distance clustering. For small corpuses,
    we use a greedy nearest-neighbor approach that doesn't require sklearn.

    Args:
        similarity_threshold: Cosine similarity threshold for grouping (0.0-1.0).
        max_cluster_size: Maximum memories per cluster before splitting.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_cluster_size: int = 10,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_cluster_size = max_cluster_size

    def cluster_by_content(
        self,
        memories: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """Cluster memories by content similarity using text hashing.

        This is a fast, deterministic clustering method that groups memories
        with similar content together without requiring embedding vectors.

        Args:
            memories: List of memory dicts with 'content' key.

        Returns:
            List of clusters (each cluster is a list of memory dicts).
        """
        if not memories:
            return []

        # Normalize content for grouping
        def normalize(text: str) -> str:
            return re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text.lower())

        # Group by normalized content hash (exact dedup clusters)
        hash_groups: Dict[str, List[Dict[str, Any]]] = {}
        for mem in memories:
            content = mem.get("content", "")
            norm = normalize(content)
            h = hashlib.sha256(norm.encode()).hexdigest()[:12]
            if h not in hash_groups:
                hash_groups[h] = []
            hash_groups[h].append(mem)

        # Merge small clusters by topic/keyword overlap
        clusters = list(hash_groups.values())

        # Further merge clusters with overlapping keywords
        merged = self._merge_by_keywords(clusters)
        return merged

    def cluster_by_vectors(
        self,
        memories: List[Dict[str, Any]],
        vectors: List[List[float]],
    ) -> List[List[Dict[str, Any]]]:
        """Cluster memories by embedding vector similarity.

        Uses greedy nearest-neighbor clustering.

        Args:
            memories: List of memory dicts.
            vectors: Corresponding embedding vectors.

        Returns:
            List of clusters.
        """
        if not memories or not vectors:
            return []

        n = len(memories)
        assigned = [False] * n
        clusters = []

        for i in range(n):
            if assigned[i]:
                continue
            cluster = [memories[i]]
            assigned[i] = True

            vi = np_array(vectors[i])
            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                vj = np_array(vectors[j])
                sim = cosine_similarity(vi, vj)
                if sim >= self.similarity_threshold and len(cluster) < self.max_cluster_size:
                    cluster.append(memories[j])
                    assigned[j] = True

            clusters.append(cluster)

        return clusters

    def _merge_by_keywords(
        self,
        clusters: List[List[Dict[str, Any]]],
    ) -> List[List[Dict[str, Any]]]:
        """Merge small clusters that share keywords."""
        if len(clusters) <= 1:
            return clusters

        merged = []
        used = set()

        for i, ci in enumerate(clusters):
            if i in used:
                continue
            keywords_i = self._extract_keywords(ci)
            combined = list(ci)

            for j in range(i + 1, len(clusters)):
                if j in used:
                    continue
                keywords_j = self._extract_keywords(clusters[j])
                overlap = len(keywords_i & keywords_j)
                # Merge if >30% keyword overlap and both small
                if (overlap > 0 and
                    overlap / max(len(keywords_i), len(keywords_j)) > 0.3 and
                    len(combined) + len(clusters[j]) <= self.max_cluster_size):
                    combined.extend(clusters[j])
                    used.add(j)

            merged.append(combined)
            used.add(i)

        return merged

    @staticmethod
    def _extract_keywords(cluster: List[Dict[str, Any]]) -> set:
        """Extract significant keywords from a cluster of memories."""
        words = set()
        for mem in cluster:
            content = mem.get("content", "")
            # Chinese entities
            for m in re.finditer(r"[\u4e00-\u9fa5]{2,8}", content):
                words.add(m.group())
            # English terms
            for m in re.finditer(r"[A-Za-z]{3,}", content):
                words.add(m.group().lower())
        return words


class SummaryGenerator:
    """Generate LLM summaries for memory clusters.

    Args:
        llm_callback: Function that takes a prompt and returns LLM text.
                       If None, uses OpenAI client from memory_settings.
    """

    def __init__(
        self,
        llm_callback: Optional[Callable[[str], str]] = None,
    ):
        self.llm_callback = llm_callback
        self._openai_client = None

        if llm_callback is None and memory_settings.llm_api_key:
            from openai import OpenAI
            self._openai_client = OpenAI(
                api_key=memory_settings.llm_api_key,
                base_url=memory_settings.llm_base_url,
            )

    def summarize_cluster(
        self,
        memories: List[Dict[str, Any]],
    ) -> str:
        """Generate a summary for a cluster of memories.

        Args:
            memories: List of memory dicts with 'content' key.

        Returns:
            Summary text string.
        """
        if not memories:
            return ""

        if len(memories) == 1:
            return memories[0].get("content", "")

        # Concatenate memories for the prompt
        memory_texts = [m.get("content", "") for m in memories]
        combined = "\n".join(f"- {t}" for t in memory_texts)

        prompt = CLUSTER_SUMMARY_PROMPT.format(memories=combined)

        if self.llm_callback:
            try:
                return self.llm_callback(prompt).strip()
            except Exception as e:
                log.warning("LLM cluster summarization failed: %s", e)
                return self._heuristic_summary(memory_texts)

        if self._openai_client:
            try:
                response = self._openai_client.chat.completions.create(
                    model=memory_settings.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=256,
                )
                if not response.choices:
                    return self._heuristic_summary(memory_texts)
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                log.warning("OpenAI cluster summarization failed: %s", e)
                return self._heuristic_summary(memory_texts)

        return self._heuristic_summary(memory_texts)

    @staticmethod
    def _heuristic_summary(texts: List[str]) -> str:
        """Simple heuristic summary: longest text + key facts from others."""
        if not texts:
            return ""
        # Use the longest text as base
        base = max(texts, key=len)
        # Extract unique sentences from other texts
        extra_facts = []
        for t in texts:
            if t == base:
                continue
            sentences = re.split(r"[。！？.!?]+", t)
            for s in sentences:
                s = s.strip()
                if len(s) > 10 and s not in base:
                    extra_facts.append(s)
        summary = base
        if extra_facts:
            summary += "。此外，" + "；".join(extra_facts[:3])
        return summary[:500]


class PruningEngine:
    """Prune low-importance / decayed memories.

    Args:
        importance_threshold: Minimum importance score to keep (0.0-1.0).
        retention_threshold: Minimum Ebbinghaus retention to keep (0.0-1.0).
        max_age_hours: Maximum age in hours before pruning (0 = no limit).
    """

    def __init__(
        self,
        importance_threshold: float = 0.3,
        retention_threshold: float = 0.2,
        max_age_hours: float = 0,
    ):
        self.importance_threshold = importance_threshold
        self.retention_threshold = retention_threshold
        self.max_age_hours = max_age_hours
        self._decay = EbbinghausDecay()
        self._importance = ImportanceEvaluator()

    def should_prune(
        self,
        memory: Dict[str, Any],
        current_time: Optional[float] = None,
    ) -> bool:
        """Determine if a memory should be pruned.

        Args:
            memory: Memory dict with keys: content, importance, created_at,
                     access_count, retention (optional).
            current_time: Current timestamp (defaults to time.time()).

        Returns:
            True if the memory should be pruned.
        """
        now = current_time or time.time()
        content = memory.get("content", "")

        # 1. Importance check
        importance = memory.get("importance")
        if importance is None:
            importance = self._importance.score(content)
        if importance < self.importance_threshold:
            return True

        # 2. Ebbinghaus decay check
        retention = memory.get("retention")
        if retention is None:
            created_at = memory.get("created_at", now)
            elapsed_hours = (now - created_at) / 3600
            access_count = memory.get("access_count", 0)
            initial = memory.get("initial_importance", importance)
            retention = self._decay.retention(initial, elapsed_hours, access_count)
        if retention < self.retention_threshold:
            return True

        # 3. Age check
        if self.max_age_hours > 0:
            created_at = memory.get("created_at", now)
            age_hours = (now - created_at) / 3600
            if age_hours > self.max_age_hours:
                return True

        return False

    def prune_batch(
        self,
        memories: List[Dict[str, Any]],
        current_time: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Prune a batch of memories.

        Args:
            memories: List of memory dicts to evaluate.
            current_time: Current timestamp.

        Returns:
            (kept, pruned) tuple of memory lists.
        """
        kept, pruned = [], []
        for mem in memories:
            if self.should_prune(mem, current_time):
                pruned.append(mem)
            else:
                kept.append(mem)
        return kept, pruned


class MemoryCompressor:
    """Full compression pipeline: cluster → summarize → archive → prune.

    This is the main entry point for memory compression, aligned with
    PowerMem's MemoryOptimizer.compress() method.

    Args:
        cluster_finder: ClusterFinder instance.
        summary_generator: SummaryGenerator instance.
        pruning_engine: PruningEngine instance.
        llm_callback: Optional LLM callback for summaries.
    """

    def __init__(
        self,
        cluster_finder: Optional[ClusterFinder] = None,
        summary_generator: Optional[SummaryGenerator] = None,
        pruning_engine: Optional[PruningEngine] = None,
        llm_callback: Optional[Callable[[str], str]] = None,
    ):
        self.cluster_finder = cluster_finder or ClusterFinder()
        self.summary_generator = summary_generator or SummaryGenerator(llm_callback=llm_callback)
        self.pruning_engine = pruning_engine or PruningEngine()

    def compress(
        self,
        memories: List[Dict[str, Any]],
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compress memories: cluster similar ones, summarize, prune low-value.

        Args:
            memories: List of memory dicts to compress.
            current_time: Current timestamp.

        Returns:
            Dict with keys:
              - summaries: list of {cluster_id, summary, source_ids}
              - kept: memories that survived pruning
              - pruned: memories that were pruned
              - archived: original memories replaced by summaries
              - stats: compression statistics
        """
        if not memories:
            return {
                "summaries": [],
                "kept": [],
                "pruned": [],
                "archived": [],
                "stats": {"input_count": 0, "output_count": 0, "compression_ratio": 0},
            }

        # Step 1: Cluster similar memories
        clusters = self.cluster_finder.cluster_by_content(memories)

        # Step 2: Generate summaries for multi-memory clusters
        summaries = []
        archived_ids = set()
        for i, cluster in enumerate(clusters):
            if len(cluster) <= 1:
                # Single-memory clusters are kept as-is
                continue

            summary_text = self.summary_generator.summarize_cluster(cluster)
            source_ids = [m.get("id", m.get("memory_id", "")) for m in cluster]
            summaries.append({
                "cluster_id": i,
                "summary": summary_text,
                "source_ids": source_ids,
                "source_count": len(cluster),
            })
            # Original memories in this cluster are archived
            for sid in source_ids:
                archived_ids.add(sid)

        # Step 3: Prune low-importance memories
        kept, pruned = self.pruning_engine.prune_batch(memories, current_time)

        # Filter out archived memories from kept list
        kept = [m for m in kept if m.get("id", m.get("memory_id", "")) not in archived_ids]

        input_count = len(memories)
        output_count = len(summaries) + len(kept)
        compression_ratio = output_count / input_count if input_count > 0 else 0

        return {
            "summaries": summaries,
            "kept": kept,
            "pruned": pruned,
            "archived": [m for m in memories if m.get("id", m.get("memory_id", "")) in archived_ids],
            "stats": {
                "input_count": input_count,
                "cluster_count": len(clusters),
                "summary_count": len(summaries),
                "kept_count": len(kept),
                "pruned_count": len(pruned),
                "archived_count": len(archived_ids),
                "output_count": output_count,
                "compression_ratio": round(compression_ratio, 2),
            },
        }


# Utility functions for vector math (no numpy dependency for pure content clustering)

def np_array(vec: List[float]) -> Any:
    """Convert list to numpy array."""
    import numpy as np
    return np.array(vec, dtype=np.float32)


def cosine_similarity(a: Any, b: Any) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
