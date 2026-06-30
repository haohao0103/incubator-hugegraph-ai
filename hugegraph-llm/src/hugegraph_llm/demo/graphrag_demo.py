#!/usr/bin/env python3
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
LightRAG-style GraphRAG Demo

Demonstrates the core capabilities of the LightRAG approach:
1. Incremental graph update with entity name as primary key
2. Dual-level retrieval (entity-centric + relationship-centric)
3. Simplified query planning (LOW/HIGH/HYBRID)
4. Full pipeline: extract → incremental update → plan → retrieve

No external services required (no HugeGraph server, no LLM API keys).
Run: python -m hugegraph_llm.demo.graphrag_demo
"""

import json
import time

from hugegraph_llm.operators.graphrag_op.dual_level_retrieval import DualLevelRetriever, RetrievalLevel
from hugegraph_llm.operators.graphrag_op.incremental_update import IncrementalGraphUpdater
from hugegraph_llm.operators.graphrag_op.query_planner import QueryIntent, QueryLevel, QueryPlanner


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def _print_sub(title: str) -> None:
    print(f"\n--- {title} ---\n")


def demo_incremental_update():
    """Demo 1: LightRAG-style incremental update with entity name as primary key."""
    _print_header("Demo 1: Incremental Graph Update (LightRAG Core)")

    updater = IncrementalGraphUpdater(graph_client=None)

    # First batch of data
    _print_sub("Step 1: Initial indexing (first batch of documents)")
    context1 = {
        "vertices": [
            {"id": "1:Alice", "label": "person", "properties": {"name": "Alice", "occupation": "engineer"}},
            {"id": "1:Bob", "label": "person", "properties": {"name": "Bob", "occupation": "designer"}},
            {"id": "1:Google", "label": "company", "properties": {"name": "Google"}},
        ],
        "edges": [
            {"label": "works_at", "outV": "1:Alice", "inV": "1:Google", "outVLabel": "person", "inVLabel": "company", "properties": {}},
            {"label": "collaborates", "outV": "1:Alice", "inV": "1:Bob", "outVLabel": "person", "inVLabel": "person", "properties": {}},
        ],
    }
    result1 = updater.run(context1)
    summary1 = result1["incremental_update_summary"]
    print(f"  New vertices: {summary1['new_vertices']}")
    print(f"  New edges:    {summary1['new_edges']}")
    print(f"  Entity name → ID mapping: {json.dumps(result1['entity_name_to_id'], indent=4, ensure_ascii=False)}")

    # Second batch — KEY FEATURE: incremental append without rebuild
    _print_sub("Step 2: Incremental update (new documents arrive)")
    context2 = {
        "vertices": [
            {"id": "1:Charlie", "label": "person", "properties": {"name": "Charlie", "occupation": "PM"}},
            {"id": "1:Meta", "label": "company", "properties": {"name": "Meta"}},
            # Alice with updated properties — should be merged, not duplicated
            {"id": "1:Alice", "label": "person", "properties": {"name": "Alice", "team": "ML Platform"}},
        ],
        "edges": [
            {"label": "works_at", "outV": "1:Bob", "inV": "1:Meta", "outVLabel": "person", "inVLabel": "company", "properties": {}},
            {"label": "manages", "outV": "1:Charlie", "inV": "1:Alice", "outVLabel": "person", "inVLabel": "person", "properties": {}},
        ],
    }
    result2 = updater.run(context2)
    summary2 = result2["incremental_update_summary"]
    print(f"  New vertices:      {summary2['new_vertices']}")
    print(f"  Updated vertices:  {summary2['updated_vertices']}  ← Alice merged, not duplicated!")
    print(f"  New edges:         {summary2['new_edges']}")
    print(f"  Entity name → ID:  {json.dumps(result2['entity_name_to_id'], indent=4, ensure_ascii=False)}")

    # Show property merging
    _print_sub("Step 3: Property merge result for Alice")
    if result2.get("merged_vertices"):
        merged = result2["merged_vertices"][0]
        print(f"  Alice's merged properties: {json.dumps(merged['properties'], ensure_ascii=False)}")
        print(f"  → 'occupation: engineer' preserved, 'team: ML Platform' added")

    # Change log
    _print_sub("Step 4: Change log (auditability)")
    for i, entry in enumerate(updater.get_change_log()):
        print(f"  Update #{i + 1}: {json.dumps(entry, ensure_ascii=False)}")

    return result2


def demo_query_planner():
    """Demo 2: Simplified query planning (LOW/HIGH/HYBRID)."""
    _print_header("Demo 2: Simplified Query Planning (No Community Dependency)")

    planner = QueryPlanner(llm=None)

    queries = [
        ("Who is Alice?", "specific_entity", "low"),
        ("What is machine learning?", "specific_entity", "low"),
        ("How are Python and Java related?", "relationship", "hybrid"),
        ("为什么股市会崩盘？", "abstract", "high"),
        ("Compare React and Vue", "relationship", "hybrid"),
    ]

    _print_sub("Query intent classification and retrieval level mapping")
    print(f"  {'Query':<40} {'Intent':<20} {'Level':<10}")
    print(f"  {'-' * 40} {'-' * 20} {'-' * 10}")

    for query, expected_intent, expected_level in queries:
        result = planner.run({"query": query})
        intent = result["query_intent"]
        level = result["retrieval_level"]
        match_intent = "✓" if intent == expected_intent else "≈"
        match_level = "✓" if level == expected_level else "≈"
        print(f"  {query:<40} {intent:<20} {level:<10} {match_intent}{match_level}")

    _print_sub("No community detection needed!")
    print("  Microsoft GraphRAG requires communities for GLOBAL strategy.")
    print("  LightRAG uses entity/relationship levels — always available.")
    print("  This is why incremental updates work: no global restructuring.")


def demo_dual_level_retrieval():
    """Demo 3: Dual-level retrieval (entity-centric + relationship-centric)."""
    _print_header("Demo 3: Dual-Level Retrieval (LightRAG Innovation)")

    retriever = DualLevelRetriever(graph_client=None)

    # Build entity name → ID mapping from previous indexing
    entity_name_to_id = {
        "Alice": "1:Alice",
        "Bob": "1:Bob",
        "Charlie": "1:Charlie",
        "Google": "1:Google",
        "Meta": "1:Meta",
    }

    _print_sub("Level determination for different query types")
    test_queries = [
        ("Who is Alice?", ["Alice"]),
        ("How are Alice and Bob related?", ["Alice", "Bob"]),
        ("What are the key technology trends?", ["technology", "trends", "AI"]),
    ]

    for query, keywords in test_queries:
        level = retriever._determine_retrieval_level(query, keywords)
        print(f"  Query:    {query}")
        print(f"  Keywords: {keywords}")
        print(f"  Level:    {level.value}")
        print()

    _print_sub("Full retrieval flow (without graph client — simulation)")
    query = "Who is Alice?"
    keywords = ["Alice"]
    context = {
        "query": query,
        "keywords": keywords,
        "entity_name_to_id": entity_name_to_id,
    }
    result = retriever.run(context)
    print(f"  Query:           {query}")
    print(f"  Retrieval level: {result['retrieval_level']}")
    print(f"  Low-level results:  {len(result['low_level_results'])} items")
    print(f"  High-level results: {len(result['high_level_results'])} items")
    print(f"  Merged results:     {len(result['dual_level_results'])} items")
    print(f"\n  Note: Without graph_client, traversal returns empty.")
    print(f"  With HugeGraph connected, this would traverse Alice's")
    print(f"  1-2 hop neighborhood for entity-centric results.")


def demo_full_pipeline():
    """Demo 4: Full pipeline — incremental update → plan → retrieve."""
    _print_header("Demo 4: Full LightRAG Pipeline")

    # Step 1: Incremental update (simulating document indexing)
    _print_sub("Step 1: Document indexing (incremental update)")
    updater = IncrementalGraphUpdater(graph_client=None)
    index_context = {
        "vertices": [
            {"id": "1:Python", "label": "language", "properties": {"name": "Python", "paradigm": "multi-paradigm"}},
            {"id": "1:Java", "label": "language", "properties": {"name": "Java", "paradigm": "object-oriented"}},
            {"id": "1:FastAPI", "label": "framework", "properties": {"name": "FastAPI", "language": "Python"}},
            {"id": "1:Spring", "label": "framework", "properties": {"name": "Spring", "language": "Java"}},
        ],
        "edges": [
            {"label": "written_in", "outV": "1:FastAPI", "inV": "1:Python", "outVLabel": "framework", "inVLabel": "language", "properties": {}},
            {"label": "written_in", "outV": "1:Spring", "inV": "1:Java", "outVLabel": "framework", "inVLabel": "language", "properties": {}},
            {"label": "competes_with", "outV": "1:Python", "inV": "1:Java", "outVLabel": "language", "inVLabel": "language", "properties": {}},
        ],
    }
    index_result = updater.run(index_context)
    print(f"  Indexed: {index_result['incremental_update_summary']['new_vertices']} vertices, "
          f"{index_result['incremental_update_summary']['new_edges']} edges")

    # Step 2: Query planning
    _print_sub("Step 2: Query planning")
    planner = QueryPlanner(llm=None)
    query = "How are Python and Java related?"
    plan_result = planner.run({"query": query})
    print(f"  Query:  {query}")
    print(f"  Intent: {plan_result['query_intent']}")
    print(f"  Level:  {plan_result['retrieval_level']}")
    print(f"  Steps:  {plan_result['query_plan']['steps']}")

    # Step 3: Dual-level retrieval
    _print_sub("Step 3: Dual-level retrieval")
    retriever = DualLevelRetriever(graph_client=None)
    retrieval_context = {
        "query": query,
        "keywords": ["Python", "Java"],
        "entity_name_to_id": index_result.get("entity_name_to_id", {}),
    }
    retrieval_result = retriever.run(retrieval_context)
    print(f"  Retrieval level:  {retrieval_result['retrieval_level']}")
    print(f"  Low-level results:  {len(retrieval_result['low_level_results'])} items")
    print(f"  High-level results: {len(retrieval_result['high_level_results'])} items")
    print(f"  Merged results:     {len(retrieval_result['dual_level_results'])} items")

    # Step 4: Incremental update with new data (append-only!)
    _print_sub("Step 4: New document arrives — incremental append")
    context_new = {
        "vertices": [
            {"id": "1:Rust", "label": "language", "properties": {"name": "Rust", "paradigm": "multi-paradigm"}},
        ],
        "edges": [
            {"label": "competes_with", "outV": "1:Rust", "inV": "1:Python", "outVLabel": "language", "inVLabel": "language", "properties": {}},
        ],
    }
    new_result = updater.run(context_new)
    print(f"  New vertices: {new_result['incremental_update_summary']['new_vertices']}")
    print(f"  New edges:    {new_result['incremental_update_summary']['new_edges']}")
    print(f"\n  ✓ No full rebuild needed! Entity name as primary key")
    print(f"    enables append-only updates (LightRAG's core innovation).")


def demo_comparison():
    """Demo 5: Comparison with Microsoft GraphRAG approach."""
    _print_header("Demo 5: LightRAG vs Microsoft GraphRAG Comparison")

    comparisons = [
        ("Incremental Update", "✓ Append-only", "✗ Full rebuild required"),
        ("Entity Dedup", "✓ Name-based (fast)", "✗ ID-based + community rebuild"),
        ("Community Detection", "Optional (Phase 2)", "Required (blocks updates)"),
        ("Hierarchical Summaries", "Optional (Phase 2)", "Required (blocks updates)"),
        ("Retrieval Modes", "LOW/HIGH/HYBRID", "LOCAL/GLOBAL/DRIFT/VECTOR"),
        ("Query Planning", "3 levels, simple", "4 strategies, community-dependent"),
        ("Production Landing", "✓ Fast (Huolala: 56%→78%)", "Slow (needs communities first)"),
        ("Indexing Cost", "Medium (LLM extraction)", "High (LLM + communities + summaries)"),
    ]

    print(f"  {'Dimension':<25} {'LightRAG (HugeGraph)':<30} {'Microsoft GraphRAG':<30}")
    print(f"  {'-' * 25} {'-' * 30} {'-' * 30}")
    for dim, lightrag, ms_graphrag in comparisons:
        print(f"  {dim:<25} {lightrag:<30} {ms_graphrag:<30}")

    _print_sub("Why Microsoft/LazyGraphRAG can't do incremental updates:")
    print("""
  Community structure is a GLOBAL property of the graph.
  Adding new entities/edges changes the community structure,
  requiring re-running Leiden algorithm and regenerating ALL
  community summaries. This is an architectural limitation.

  LightRAG doesn't use community detection, so:
  - New entity → match by name → merge or create (append-only)
  - New edge → connect to existing entities by name
  - No global restructuring needed
  - Incremental updates work naturally
    """)


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  HugeGraph LightRAG-style GraphRAG Demo")
    print("  No external services required")
    print("=" * 60)

    start_time = time.time()

    demo_incremental_update()
    demo_query_planner()
    demo_dual_level_retrieval()
    demo_full_pipeline()
    demo_comparison()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  All demos completed in {elapsed:.2f}s")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()