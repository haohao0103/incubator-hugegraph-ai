"""
End-to-End LLM Evaluation for Community Detection
===================================================

Uses MiMo LLM to evaluate GraphRAG quality with real QA tasks.

Pipeline:
1. Load graph + community partitions (Louvain & Leiden)
2. Generate QA pairs from community structure
3. Simulate retrieval (global vs local)
4. Generate answers with MiMo
5. Judge answer quality with MiMo
6. Compare Leiden vs Louvain

Usage:
    python e2e_llm_eval.py
"""

import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_FILE = BASE_DIR / "e2e_llm_results.json"

# ── MiMo LLM Client ──────────────────────────────────────────

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.xiaomimimo.com/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = "mimo-v2.5-pro"


def _get_llm_text(response) -> str:
    """Extract text from MiMo reasoning model response."""
    msg = response.choices[0].message
    if msg.content:
        return msg.content
    rc = getattr(msg, "reasoning_content", None)
    if rc:
        return rc
    return ""


def call_llm(prompt: str, max_tokens: int = 512) -> str:
    """Call MiMo LLM with retry."""
    from openai import OpenAI
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_completion_tokens=max_tokens,
        )
        return _get_llm_text(response).strip()
    except Exception as e:
        print(f"[LLM Error] {e}")
        return ""


# ── Data Loading ─────────────────────────────────────────────

def load_edge_list(path: Path) -> nx.Graph:
    G = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    return G


def load_partition(path: Path) -> Dict[int, int]:
    partition = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                partition[int(parts[0])] = int(parts[1])
    return partition


def communities_from_partition(partition: Dict[int, int]) -> Dict[int, List[int]]:
    comms = defaultdict(list)
    for node, comm in partition.items():
        comms[comm].append(node)
    return dict(comms)


# ── QA Pair Generation ───────────────────────────────────────

def generate_qa_pairs(G: nx.Graph, partition: Dict[int, int], num_pairs: int = 10) -> List[Dict]:
    """Generate QA pairs from graph structure."""
    comms = communities_from_partition(partition)
    nodes = list(G.nodes())
    edges = list(G.edges())
    qa_pairs = []

    # Global QA: community-level
    comm_ids = list(comms.keys())
    for i, comm_id in enumerate(comm_ids[:num_pairs // 2]):
        members = comms[comm_id]
        sample = members[:5]
        q = f"Community {comm_id} contains nodes: {', '.join(map(str, sample))}. " \
            f"How many nodes are in this community?"
        a = str(len(members))
        qa_pairs.append({
            "id": f"global_{i}",
            "type": "global",
            "question": q,
            "answer": a,
            "comm_id": comm_id,
        })

    # Local QA: entity-level
    for i in range(num_pairs // 2):
        node = nodes[i % len(nodes)]
        neighbors = list(G.neighbors(node))
        q = f"What is the degree (number of neighbors) of node {node}?"
        a = str(len(neighbors))
        qa_pairs.append({
            "id": f"local_{i}",
            "type": "local",
            "question": q,
            "answer": a,
            "node": node,
        })

    return qa_pairs


# ── Context Retrieval ────────────────────────────────────────

def retrieve_global_context(G: nx.Graph, partition: Dict[int, int], qa: Dict) -> str:
    """Retrieve community summary for global QA."""
    comms = communities_from_partition(partition)
    comm_id = qa.get("comm_id", 0)
    members = comms.get(comm_id, [])
    return f"Community {comm_id} has {len(members)} nodes. " \
           f"Sample members: {', '.join(map(str, members[:10]))}."


def retrieve_local_context(G: nx.Graph, partition: Dict[int, int], qa: Dict) -> str:
    """Retrieve 1-hop subgraph for local QA."""
    node = qa.get("node", 0)
    neighbors = list(G.neighbors(node))
    return f"Node {node} has {len(neighbors)} neighbors: {', '.join(map(str, neighbors[:10]))}."


# ── Answer Generation ────────────────────────────────────────

def generate_answer(question: str, context: str) -> str:
    prompt = f"""Based on the following context, answer the question concisely.

Context: {context}

Question: {question}

Answer:"""
    return call_llm(prompt, max_tokens=128)


# ── LLM-as-Judge ─────────────────────────────────────────────

def judge_answer(question: str, ground_truth: str, generated: str) -> Dict:
    prompt = f"""You are an evaluator. Rate the generated answer against the ground truth.

Question: {question}
Ground Truth: {ground_truth}
Generated Answer: {generated}

Rate on a scale of 1-5:
- 5: Correct and complete
- 4: Correct but slightly incomplete
- 3: Partially correct
- 2: Mostly incorrect
- 1: Completely wrong or irrelevant

Respond ONLY with JSON: {{"score": X, "reason": "brief explanation"}}"""

    response = call_llm(prompt, max_tokens=256)
    # Extract JSON
    import re
    m = re.search(r'\{[^}]+\}', response)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    # Fallback: extract score
    score_match = re.search(r'["\']?score["\']?\s*[:=]\s*(\d)', response)
    score = int(score_match.group(1)) if score_match else 3
    return {"score": score, "reason": response[:100]}


# ── Main Evaluation ──────────────────────────────────────────

def evaluate_algorithm(name: str, G: nx.Graph, partition: Dict[int, int], qa_pairs: List[Dict]) -> Dict:
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    total_score = 0
    results = []

    for qa in qa_pairs:
        # Retrieve context
        if qa["type"] == "global":
            context = retrieve_global_context(G, partition, qa)
        else:
            context = retrieve_local_context(G, partition, qa)

        # Generate answer
        generated = generate_answer(qa["question"], context)

        # Judge
        judgment = judge_answer(qa["question"], qa["answer"], generated)
        score = judgment.get("score", 3)
        total_score += score

        results.append({
            "qa_id": qa["id"],
            "type": qa["type"],
            "question": qa["question"],
            "ground_truth": qa["answer"],
            "generated": generated,
            "score": score,
            "reason": judgment.get("reason", ""),
        })

        print(f"  [{qa['type']:6}] Q: {qa['question'][:50]}... Score: {score}/5")

    avg_score = total_score / len(qa_pairs) if qa_pairs else 0
    print(f"\n  Average Score: {avg_score:.2f}/5.0")

    return {
        "algorithm": name,
        "num_qa": len(qa_pairs),
        "avg_score": avg_score,
        "details": results,
    }


def main():
    print("=" * 60)
    print("GraphRAG E2E LLM Evaluation (MiMo)")
    print("=" * 60)

    # Load LFR small easy
    edge_file = DATA_DIR / "lfr_small_easy_edges.txt"
    louvain_file = DATA_DIR / "lfr_small_easy_partition_louvain.txt"
    leiden_file = DATA_DIR / "lfr_small_easy_partition_leiden.txt"

    if not edge_file.exists() or not louvain_file.exists() or not leiden_file.exists():
        print("ERROR: Dataset files not found. Run evaluate_graphrag_quality.py first.")
        sys.exit(1)

    G = load_edge_list(edge_file)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    louvain_partition = load_partition(louvain_file)
    leiden_partition = load_partition(leiden_file)

    # Generate QA pairs (using Louvain as reference)
    qa_pairs = generate_qa_pairs(G, louvain_partition, num_pairs=10)
    print(f"Generated {len(qa_pairs)} QA pairs")

    # Evaluate Louvain
    louvain_result = evaluate_algorithm("louvain", G, louvain_partition, qa_pairs)

    # Evaluate Leiden
    leiden_result = evaluate_algorithm("leiden", G, leiden_partition, qa_pairs)

    # Save results
    final = {
        "dataset": "lfr_small_easy",
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "louvain": louvain_result,
        "leiden": leiden_result,
        "winner": "leiden" if leiden_result["avg_score"] > louvain_result["avg_score"] else "louvain",
        "delta": leiden_result["avg_score"] - louvain_result["avg_score"],
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(final, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Louvain: {louvain_result['avg_score']:.2f}/5.0")
    print(f"Leiden:  {leiden_result['avg_score']:.2f}/5.0")
    print(f"Winner:  {final['winner']} (Δ = {final['delta']:+.2f})")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
