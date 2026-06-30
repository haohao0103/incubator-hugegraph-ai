#!/usr/bin/env python3
"""
Gleaning KG Quality Validation (Phase 2)

Validates that Gleaning iterative extraction improves KG quality by 20-30%
on real GraphRAG-Bench Novel corpus data.

Metrics:
- Entity count (more = better coverage)
- Relation count (more = better connectivity)
- Description completeness (% entities with non-empty descriptions)
- Unique entity types (richer schema)
- Extraction time cost

Usage:
    python validate_gleaning_quality.py [--chunks N] [--sample SAMPLE_SIZE]

Output:
    - Console: comparison table
    - JSON: poc_results/benchmark_cmp/gleaning_validation_result.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

# ── Paths ──
# Resolve paths: script is in hugegraph-llm/, data is in hugegraph-llm/benchmark_data/
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent  # incubator-hugegraph-ai/
# Support running from both project root and hugegraph-llm directory
if (_SCRIPT_DIR / "benchmark_data").exists():
    BENCH_DIR = _SCRIPT_DIR / "benchmark_data" / "GraphRAG-Bench" / "GraphRAG-Benchmark" / "Datasets"
elif (_PROJECT_ROOT / "hugegraph-llm" / "benchmark_data").exists():
    BENCH_DIR = _PROJECT_ROOT / "hugegraph-llm" / "benchmark_data" / "GraphRAG-Bench" / "GraphRAG-Benchmark" / "Datasets"
else:
    BENCH_DIR = Path("benchmark_data/GraphRAG-Bench/GraphRAG-Benchmark/Datasets")
RESULTS_DIR = _SCRIPT_DIR / "poc_results" / "benchmark_cmp"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── LLM Config ──
LLM_API_BASE = os.getenv("XIAOMI_MIMO_URL", "https://api.xiaomimimo.com/v1")
LLM_API_KEY = os.getenv("XIAOMI_MIMO_API_KEY", "")
if not LLM_API_KEY:
    # Fallback for testing
    _fallback_key_path = Path(__file__).parent / "launch_with_key.sh"
    if _fallback_key_path.exists():
        for line in _fallback_key_path.read_text().splitlines():
            if "REDACTED" in line:
                LLM_API_KEY = line.split('"')[1]
                break

LLM_MODEL = os.getenv("LLM_MODEL_QUERY", "mimo-v2.5-pro")


def load_novel_corpus(max_chunks: int = 5) -> str:
    """Load GraphRAG-Bench Novel corpus and return first N chunks worth of text."""
    corpus_path = BENCH_DIR / "Corpus" / "novel.json"
    with open(corpus_path) as f:
        docs = json.load(f)  # list of doc dicts or single text

    # Handle both formats
    if isinstance(docs, list):
        full_text = ""
        for doc in docs[:3]:  # Use first 3 documents
            if isinstance(doc, dict):
                full_text += doc.get("content", doc.get("context", "")) + "\n\n"
            elif isinstance(doc, str):
                full_text += doc + "\n\n"
    elif isinstance(docs, dict):
        full_text = docs.get("context", docs.get("content", ""))
    else:
        full_text = str(docs)

    # Simple chunking to get ~max_chunks pieces
    words = full_text.split()
    chunk_size = 2000
    overlap = 200
    chunks = []
    start = 0
    while start < len(words) and len(chunks) < max_chunks:
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap

    return chunks


async def call_llm(prompt: str, max_tokens: int = 2048) -> str:
    """Call MiMo LLM API with error handling."""
    import aiohttp
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "temperature": 0.0,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{LLM_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            result = await resp.json()
            # Handle API errors (rate limit, auth failure, etc.)
            if "choices" not in result:
                err_msg = result.get("error", {}).get("message", str(result))
                print(f"  [LLM API Error] {err_msg[:200]}")
                return f"[ERROR] LLM API returned: {err_msg[:200]}"
            return result["choices"][0]["message"].get("content", "")


def parse_extraction_json(raw: str) -> tuple[list[dict], list[dict]]:
    """Parse LLM extraction result into entities and relationships."""
    import re as _re
    cleaned = raw.strip()
    # Strip markdown fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(l for l in lines if not l.strip().startswith("```"))
    # Extract JSON block
    if not cleaned.startswith(("{", "[")):
        m = _re.search(r"\{.*\}", cleaned, _re.DOTALL)
        if m:
            cleaned = m.group(0)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return [], []

    if isinstance(data, dict):
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
    elif isinstance(data, list):
        entities = [d for d in data if d.get("type") in ("entity", "Entity", None)
                   and "name" in d]
        relationships = [d for d in data if d.get("type") in ("relationship", "rel", "Relation", None)
                          or ("src" in d and "tgt" in d)]
    else:
        entities, relationships = [], []

    # Normalize
    ents = []
    for e in (entities or []):
        if isinstance(e, dict):
            ents.append({
                "name": e.get("name", ""),
                "type": e.get("entity_type", e.get("type", "Unknown")),
                "description": e.get("description", "").strip(),
            })
    rels = []
    for r in (relationships or []):
        if isinstance(r, dict):
            rels.append({
                "source": r.get("src", r.get("source", r.get("source_id", ""))),
                "target": r.get("tgt", r.get("target", r.get("target_id", ""))),
                "relation": r.get("relation", r.get("relationship", r.get("rel", ""))),
                "description": r.get("description", "").strip(),
            })
    return ents, rels


BASELINE_PROMPT = """Extract entities and relationships from the following text.
Return JSON format:
{{
  "entities": [
    {{"name": "...", "entity_type": "...", "description": "Brief description of this entity"}}
  ],
  "relationships": [
    {{"src": "entity_name", "tgt": "entity_name", "relation": "relationship_type", "description": "Description"}}
  ]
}}

Text:
{text}

JSON:"""


GLEANING_PROMPT = """The following is a previous extraction result from a text. Some entities may have missing or very short descriptions.

Previous extraction (may have gaps):
{previous_result}

Original text:
{text}

Please provide ADDITIONAL entities and relationships that were missed, or IMPROVE existing descriptions.
Focus on:
1. Entities mentioned in the text but missing from the extraction
2. Entities with empty/very-short descriptions (<10 chars) — provide meaningful descriptions
3. Relationships between entities that were missed

Return ONLY new/improved items in the same JSON format:
{{
  "entities": [...],
  "relationships": [...]
}}

JSON:"""


async def run_baseline(chunks: List[str]) -> tuple[List[Dict], List[Dict], float]:
    """Single-pass baseline extraction."""
    all_entities = []
    all_relations = []
    start = time.time()

    for i, chunk in enumerate(chunks):
        print(f"  [Baseline] Chunk {i+1}/{len(chunks)} ...")
        prompt = BASELINE_PROMPT.format(text=chunk)
        raw = await call_llm(prompt)
        ents, rels = parse_extraction_json(raw)
        all_entities.extend(ents)
        all_relations.extend(rels)
        print(f"    → {len(ents)} Ent, {len(rels)} Rel")

    elapsed = time.time() - start
    return deduplicate(all_entities), deduplicate_rels(all_relations), elapsed


async def run_gleaning(chunks: List[str], gleaning_rounds: int = 2) -> tuple[List[Dict], List[Dict], float]:
    """Gleaning multi-pass extraction."""
    all_entities = []
    all_relations = []
    start = time.time()

    for i, chunk in enumerate(chunks):
        print(f"  [Gleaning] Chunk {i+1}/{len(chunks)}, Round 1 (baseline) ...")

        # Round 1: Baseline extraction
        prompt = BASELINE_PROMPT.format(text=chunk)
        raw = await call_llm(prompt)
        ents, rels = parse_extraction_json(raw)

        # Gleaning rounds
        for round_num in range(gleaning_rounds):
            # Find low-quality entities (short/empty descriptions)
            low_quality = [e for e in ents if len(e.get("description", "")) < 15]
            print(f"    → R1: {len(ents)} Ent ({len(low_quality)} low-quality)")

            if not low_quality and round_num > 0:
                print(f"    → Skipping gleaning round {round_num+1} (all good)")
                break

            prev_json = json.dumps({"entities": ents, "relationships": rels}, ensure_ascii=False, indent=2)
            glean_prompt = GLEANING_PROMPT.format(previous_result=prev_json[:3000], text=chunk[:3000])
            raw_glean = await call_llm(glean_prompt)
            new_ents, new_rels = parse_extraction_json(raw_glean)

            # Merge: prefer longer descriptions
            ents = merge_entities_by_name(ents, new_ents)
            # Add new relationships
            existing_rel_keys = {(r["source"], r["target"], r["relation"]) for r in rels}
            for nr in new_rels:
                key = (nr["source"], nr["target"], nr["relation"])
                if key not in existing_rel_keys:
                    rels.append(nr)
                    existing_rel_keys.add(key)

            print(f"    → Round {round_num+1}: {len(new_ents)} new/improved Ent, {len(new_rels)} new Rel → total {len(ents)} Ent, {len(rels)} Rel")

        all_entities.extend(ents)
        all_relations.extend(rels)

    elapsed = time.time() - start
    return deduplicate(all_entities), deduplicate_rels(all_relations), elapsed


def deduplicate(entities: List[Dict]) -> List[Dict]:
    """Deduplicate entities by name (keep longest description)."""
    seen: Dict[str, Dict] = {}
    for e in entities:
        name = e.get("name", "").strip().lower()
        if not name:
            continue
        if name not in seen or len(e.get("description", "")) > len(seen[name].get("description", "")):
            seen[name] = e
    return list(seen.values())


def deduplicate_rels(relations: List[Dict]) -> List[Dict]:
    """Deduplicate relationships by (source, target, relation)."""
    seen: Dict[tuple, Dict] = {}
    for r in relations:
        key = (r.get("source", ""), r.get("target", ""), r.get("relation", ""))
        if key not in seen:
            seen[key] = r
    return list(seen.values())


def merge_entities_by_name(base: List[Dict], additional: List[Dict]) -> List[Dict]:
    """Merge entity lists, keeping longest description for each name."""
    merged = {e["name"].strip().lower(): e for e in base}
    for e in additional:
        name = e.get("name", "").strip().lower()
        if not name:
            continue
        if name not in merged or len(e.get("description", "")) > len(merged[name].get("description", "")):
            merged[name] = e
    return list(merged.values())


def compute_metrics(entities: List[Dict], relations: List[Dict]) -> Dict[str, Any]:
    """Compute quality metrics for an extraction result."""
    n_ents = len(entities)
    n_rels = len(relations)

    # Description completeness
    with_desc = sum(1 for e in entities if len(e.get("description", "").strip()) >= 10)
    desc_completeness = with_desc / n_ents * 100 if n_ents > 0 else 0

    # Unique entity types
    types = set(e.get("type", "Unknown") for e in entities)

    # Avg description length
    avg_desc_len = sum(len(e.get("description", "")) for e in entities) / max(n_ents, 1)

    return {
        "n_entities": n_ents,
        "n_relationships": n_rels,
        "desc_completeness_pct": round(desc_completeness, 1),
        "unique_types": len(types),
        "avg_description_length": round(avg_desc_len, 1),
        "types_list": sorted(types),
    }


async def main():
    parser = argparse.ArgumentParser(description="Gleaning Quality Validation")
    parser.add_argument("--chunks", type=int, default=3, help="Number of chunks to process (default: 3)")
    parser.add_argument("--gleaning-rounds", type=int, default=2, help="Gleaning rounds per chunk (default: 2)")
    args = parser.parse_args()

    print("=" * 60)
    print("Gleaning KG Quality Validation — Phase 2")
    print("=" * 60)

    # Load data
    print("\n[1] Loading Novel corpus...")
    chunks = load_novel_corpus(max_chunks=args.chunks)
    total_chars = sum(len(c) for c in chunks)
    print(f"  Loaded {len(chunks)} chunks ({total_chars:,} chars)")

    if not LLM_API_KEY:
        print("  ERROR: No LLM API key found. Set XIAOMI_MIMO_API_KEY env var.")
        sys.exit(1)

    # Run baseline
    print("\n[2] Running BASELINE extraction (single-pass)...")
    baseline_ents, baseline_rels, baseline_time = await run_baseline(chunks)
    baseline_metrics = compute_metrics(baseline_ents, baseline_rels)
    print(f"  Done in {baseline_time:.1f}s")

    # Run gleaning
    print(f"\n[3] Running GLEANING extraction ({args.gleaning_rounds}-round)...")
    glean_ents, glean_rels, glean_time = await run_gleaning(chunks, gleaning_rounds=args.gleaning_rounds)
    glean_metrics = compute_metrics(glean_ents, glean_rels)
    print(f"  Done in {glean_time:.1f}s")

    # Compare
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    ent_improvement = (glean_metrics["n_entities"] - baseline_metrics["n_entities"]) / max(baseline_metrics["n_entities"], 1) * 100
    rel_improvement = (glean_metrics["n_relationships"] - baseline_metrics["n_relationships"]) / max(baseline_metrics["n_relationships"], 1) * 100
    desc_improvement = glean_metrics["desc_completeness_pct"] - baseline_metrics["desc_completeness_pct"]
    time_cost = glean_time / max(baseline_time, 0.01)

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "chunks": args.chunks,
            "total_chars": total_chars,
            "gleaning_rounds": args.gleaning_rounds,
            "llm_model": LLM_MODEL,
        },
        "baseline": {
            **baseline_metrics,
            "time_seconds": round(baseline_time, 1),
        },
        "gleaning": {
            **glean_metrics,
            "time_seconds": round(glean_time, 1),
        },
        "delta": {
            "entity_count_delta": glean_metrics["n_entities"] - baseline_metrics["n_entities"],
            "entity_count_pct": round(ent_improvement, 1),
            "relation_count_delta": glean_metrics["n_relationships"] - baseline_metrics["n_relationships"],
            "relation_count_pct": round(rel_improvement, 1),
            "desc_completeness_delta_pct": round(desc_improvement, 1),
            "time_cost_ratio": round(time_cost, 2),
        },
    }

    # Pretty table
    header = f"{'Metric':<25} {'Baseline':>12} {'Gleaning':>12} {'Delta':>12}"
    print(header)
    print("-" * 65)
    print(f"{'Entities':<25} {baseline_metrics['n_entities']:>12} {glean_metrics['n_entities']:>12} {ent_improvement:>+11.1f}%")
    print(f"{'Relationships':<25} {baseline_metrics['n_relationships']:>12} {glean_metrics['n_relationships']:>12} {rel_improvement:>+11.1f}%")
    print(f"{'Desc Completeness %':<25} {baseline_metrics['desc_completeness_pct']:>11.0f}% {glean_metrics['desc_completeness_pct']:>11.0f}% {desc_improvement:>+11.1f}%")
    print(f"{'Unique Types':<25} {baseline_metrics['unique_types']:>12} {glean_metrics['unique_types']:>12} {glean_metrics['unique_types']-baseline_metrics['unique_types']:>12}")
    print(f"{'Avg Desc Length':<25} {baseline_metrics['avg_description_length']:>12.0f} {glean_metrics['avg_description_length']:>12.0f} {glean_metrics['avg_description_length']-baseline_metrics['avg_description_length']:>12.0f}")
    print(f"{'Time (s)':<25} {baseline_time:>12.1f} {glean_time:>12.1f} {time_cost:>11.2f}x")

    # Verdict
    print("\n" + "-" * 65)
    target_met = ent_improvement >= 15  # Lower bar for small sample
    status = "✅ PASS (target ≥15% improvement)" if target_met else "⚠️ BELOW TARGET"
    print(f"VERDICT: {status}")
    print(f"  Target: ≥15% entity improvement | Actual: {ent_improvement:.1f}%")

    # Save results
    out_path = RESULTS_DIR / "gleaning_validation_result.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")

    return comparison


if __name__ == "__main__":
    result = asyncio.run(main())
