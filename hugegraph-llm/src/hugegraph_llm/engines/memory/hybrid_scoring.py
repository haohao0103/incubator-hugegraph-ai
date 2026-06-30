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
Hybrid Retrieval Scoring — aligned with mem0's scoring.py.

mem0 uses additive scoring: semantic + BM25 (sigmoid-normalized) + entity boost.
PowerMem uses OceanBase hybrid_search (built-in vector+fts+sparse fusion).

Our old approach used Reciprocal Rank Fusion (RRF) which is rank-based and
loses score magnitude information. The mem0 additive scoring approach preserves
the actual similarity scores, giving better ranking for hybrid retrieval.

Key improvements:
  1. BM25 sigmoid normalization (raw BM25 scores → [0, 1])
  2. Entity boost: memories linked to query entities get +0.5 per match
  3. Additive fusion: semantic_score + bm25_score + entity_boost
  4. Adaptive sigmoid parameters based on query length
  5. Threshold filtering at the semantic level (not post-fusion)
"""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from hugegraph_llm.utils.log import log

# ── Constants ────────────────────────────────────────────────

ENTITY_BOOST_WEIGHT = 0.5  # Per-entity match boost (mem0 uses 0.5)
MAX_SEMANTIC = 1.0
MAX_SEMANTIC_BM25 = 2.0
MAX_SEMANTIC_BM25_ENTITY = 2.5


# ── BM25 Normalization ───────────────────────────────────────


def get_bm25_params(query: str) -> Tuple[float, float]:
    """Compute adaptive sigmoid parameters for BM25 normalization.

    Short queries need steep sigmoid (narrow score band), long queries
    need gentle sigmoid (wider score distribution).

    Returns: (midpoint, steepness)
    """
    word_count = len(query)  # Use character count for Chinese text
    if word_count <= 4:
        midpoint, steepness = 10.0, 1.0
    elif word_count <= 12:
        midpoint, steepness = 15.0, 0.7
    else:
        midpoint, steepness = 20.0, 0.5
    return midpoint, steepness


def normalize_bm25(raw_score: float, midpoint: float, steepness: float) -> float:
    """Sigmoid normalization: raw BM25 score → [0, 1].

    Formula: 1 / (1 + exp(-steepness * (raw_score - midpoint)))

    This replaces our old raw BM25 scores which had arbitrary magnitude
    and couldn't be meaningfully combined with FAISS cosine similarity.
    """
    if raw_score <= 0:
        return 0.0
    try:
        normalized = 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))
    except OverflowError:
        normalized = 0.0 if raw_score < midpoint else 1.0
    return round(normalized, 4)


# ── Entity Boost ─────────────────────────────────────────────


def compute_entity_boosts(
    query_entities: List[str],
    memory_entities: Dict[str, List[str]],
    boost_weight: float = ENTITY_BOOST_WEIGHT,
) -> Dict[str, float]:
    """Compute entity boost scores for each memory.

    For each query entity that appears in a memory's entity list,
    add boost_weight to that memory's score.

    Args:
        query_entities: Entity names extracted from the query
        memory_entities: Map of memory_id → list of entity names in that memory
        boost_weight: Score to add per entity match (default 0.5)

    Returns:
        Map of memory_id → entity boost score
    """
    boosts: Dict[str, float] = {}
    qe_set = set(e.lower() for e in query_entities)
    for mid, entities in memory_entities.items():
        ent_set = set(e.lower() for e in entities)
        overlap = len(qe_set & ent_set)
        if overlap > 0:
            boosts[mid] = boost_weight * overlap
    return boosts


# ── Additive Hybrid Scoring ──────────────────────────────────


def score_and_rank(
    semantic_results: List[Dict[str, Any]],
    bm25_scores: Dict[str, float],
    entity_boosts: Dict[str, float],
    threshold: float = 0.1,
    top_k: int = 10,
    explain: bool = False,
) -> List[Dict[str, Any]]:
    """Additive hybrid scoring (mem0-style).

    Score = (semantic + bm25_normalized + entity_boost) / max_possible

    max_possible depends on which channels are active:
      - Semantic only: 1.0
      - Semantic + BM25: 2.0
      - Semantic + BM25 + Entity: 2.5

    Threshold filtering is applied at the semantic level: if semantic
    score < threshold, the result is discarded regardless of BM25/entity.

    Args:
        semantic_results: List of dicts with keys: id, content, score, metadata
        bm25_scores: Map of memory_id → raw BM25 score
        entity_boosts: Map of memory_id → entity boost score
        threshold: Minimum semantic score to keep (0.1 default, mem0-style)
        top_k: Number of results to return
        explain: If True, include score breakdown in results

    Returns:
        Ranked list of result dicts with final score and optional breakdown
    """
    has_bm25 = len(bm25_scores) > 0
    has_entity = len(entity_boosts) > 0

    max_possible = MAX_SEMANTIC
    if has_bm25:
        max_possible = MAX_SEMANTIC_BM25
    if has_entity:
        max_possible = MAX_SEMANTIC_BM25_ENTITY

    # Compute midpoint/steepness from the average query length
    # (approximation: use the content of the top semantic result)
    query_approx = semantic_results[0].get("content", "") if semantic_results else ""
    midpoint, steepness = get_bm25_params(query_approx)

    results = []
    for sr in semantic_results:
        mid = sr.get("id", "")
        semantic = sr.get("score", 0.0)

        # Threshold filtering at semantic level
        if semantic < threshold:
            continue

        bm25_norm = normalize_bm25(
            bm25_scores.get(mid, 0.0), midpoint, steepness
        ) if has_bm25 else 0.0

        entity_boost = entity_boosts.get(mid, 0.0) if has_entity else 0.0

        combined = semantic + bm25_norm + entity_boost
        final_score = round(combined / max_possible, 4)

        result = {
            "id": mid,
            "content": sr.get("content", ""),
            "score": final_score,
            "metadata": sr.get("metadata", {}),
        }

        if explain:
            result["score_breakdown"] = {
                "semantic": round(semantic, 4),
                "bm25_normalized": round(bm25_norm, 4),
                "entity_boost": round(entity_boost, 4),
                "combined": round(combined, 4),
                "max_possible": max_possible,
            }

        results.append(result)

    # Sort by final score
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


# ── Query Entity Extraction (for entity boost) ───────────────


def extract_query_entities_simple(query: str) -> List[str]:
    """Quick entity extraction from query for entity boost scoring.

    Uses simple regex patterns, not the full EntityExtractor which may
    call LLM. This is fast enough for the scoring phase.
    """
    entities = []

    # Chinese org/location suffixes
    for m in re.finditer(
        r"([\u4e00-\u9fa5]{2,8})(?:公司|集团|学校|银行|医院|厂|团队|部门|市|省|区|县|路|街)",
        query,
    ):
        entities.append(m.group(1))

    # Chinese names
    for m in re.finditer(r"[\u4e00-\u9fa5]{2,4}", query):
        c = m.group(0)
        if len(c) >= 2 and c not in {
            "什么", "怎么", "哪里", "哪个", "多少", "如何", "是谁",
            "是否", "有没有", "能不能", "为什么", "的人", "那个", "这个",
            "的了在是有和也都", "不", "没", "的", "了", "在",
        }:
            entities.append(c)

    # English names
    for m in re.finditer(r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+", query):
        entities.append(m.group(0))

    # Deduplicate
    return list(set(entities))
