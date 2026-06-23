"""
MiMo LLM End-to-End Community Detection Quality Assessment
============================================================

Real LLM-based evaluation of GraphRAG community detection quality.

Pipeline:
1. Load graph + community partition (Leiden vs Louvain)
2. Generate QA pairs from community structure
3. Simulate GraphRAG retrieval (global = community summaries, local = entity subgraph)
4. Generate answers with MiMo
5. Judge answer quality with MiMo (1-5 scale)
6. Compare Leiden vs Louvain end-to-end scores

Usage:
    PYTHONPATH=src python evaluate_llm_end_to_end.py
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import networkx as nx

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_FILE = BASE_DIR / "llm_e2e_results.json"

# MiMo API config
LLM_BASE_URL = "https://api.xiaomimimo.com/v1"
LLM_API_KEY = "sk-cs5kqi80f6upqy2e3k3xi39jtizhpgf6dkdd3j9ysoupfw7p"
LLM_MODEL = "mimo-v2.5-pro"


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


def get_communities(partition: Dict[int, int]) -> Dict[int, List[int]]:
    comms = defaultdict(list)
    for node, comm in partition.items():
        comms[comm].append(node)
    return dict(comms)


# ============================================================================
# QA Pair Generation
# ============================================================================

def generate_qa_pairs(G: nx.Graph, partition: Dict[int, int], num_pairs: int = 10) -> List[Dict]:
    """Generate QA pairs from graph structure."""
    comms = get_communities(partition)
    qa_pairs = []

    # Type 1: Entity relationship questions (local)
    edges = list(G.edges())
    for i in range(min(num_pairs // 3, len(edges))):
        u, v = edges[i]
        comm_u = partition.get(u, -1)
        comm_v = partition.get(v, -1)
        same_comm = comm_u == comm_v
        qa_pairs.append({
            "question": f"What is the relationship between entity {u} and entity {v}?",
            "ground_truth": f"Entity {u} and entity {v} are connected in the graph. They {'belong to the same community' if same_comm else 'belong to different communities'}.",
            "type": "local",
            "entities": [u, v],
            "expected_communities": list(set([comm_u, comm_v])),
        })

    # Type 2: Community membership questions (global/local hybrid)
    for comm_id, members in list(comms.items())[:num_pairs // 3]:
        if len(members) < 3:
            continue
        sample = members[:3]
        qa_pairs.append({
            "question": f"Which entities are in community {comm_id}?",
            "ground_truth": f"Community {comm_id} contains entities: {', '.join(map(str, sample))} and {len(members) - 3} others.",
            "type": "global",
            "expected_communities": [comm_id],
        })

    # Type 3: Structural questions (local)
    nodes = list(G.nodes())
    for i in range(min(num_pairs // 3, len(nodes))):
        node = nodes[i]
        neighbors = list(G.neighbors(node))
        comm = partition.get(node, -1)
        qa_pairs.append({
            "question": f"What entities are connected to entity {node}?",
            "ground_truth": f"Entity {node} is connected to {len(neighbors)} entities: {', '.join(map(str, neighbors[:5]))}{'...' if len(neighbors) > 5 else ''}.",
            "type": "local",
            "entities": [node],
            "expected_communities": [comm],
        })

    return qa_pairs[:num_pairs]


# ============================================================================
# MiMo LLM Client
# ============================================================================

def call_mimo(messages: List[Dict], max_tokens: int = 512) -> str:
    """Call MiMo API."""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=max_tokens,
        )
        # Handle reasoning model: content may be empty, use reasoning_content
        msg = response.choices[0].message
        text = msg.content or ""
        if not text.strip() and hasattr(msg, "reasoning_content"):
            text = msg.reasoning_content or ""
        return text.strip()
    except Exception as e:
        print(f"[LLM Error] {e}")
        return ""


# ============================================================================
# Simulated GraphRAG Retrieval
# ============================================================================

def simulate_global_search(question: str, G: nx.Graph, partition: Dict[int, int],
                           qa: Dict) -> str:
    """Simulate global search: retrieve community summaries."""
    comms = get_communities(partition)
    expected = qa.get("expected_communities", [])

    # Build community descriptions
    descriptions = []
    for comm_id in expected[:3]:
        members = comms.get(comm_id, [])
        if not members:
            continue
        # Count internal edges
        member_set = set(members)
        internal_edges = sum(1 for u, v in G.edges() if u in member_set and v in member_set)
        descriptions.append(
            f"Community {comm_id}: {len(members)} entities, {internal_edges} internal connections. "
            f"Sample entities: {', '.join(map(str, members[:5]))}."
        )

    return "\n".join(descriptions) if descriptions else "No relevant communities found."


def simulate_local_search(question: str, G: nx.Graph, partition: Dict[int, int],
                          qa: Dict) -> str:
    """Simulate local search: retrieve entity subgraph."""
    entities = qa.get("entities", [])
    if not entities:
        return "No specific entities mentioned."

    # Build subgraph around entities
    subgraph_nodes = set(entities)
    for e in entities:
        subgraph_nodes.update(G.neighbors(e))

    edges_info = []
    for u, v in G.edges():
        if u in subgraph_nodes and v in subgraph_nodes:
            edges_info.append(f"Entity {u} -- connected to --> Entity {v}")
            if len(edges_info) >= 20:
                break

    return "\n".join(edges_info) if edges_info else "No connections found."


# ============================================================================
# Answer Generation & Judging
# ============================================================================

def generate_answer(question: str, context: str, scope: str) -> str:
    """Generate answer using MiMo."""
    prompt = f"""You are a knowledge graph assistant. Answer the question based on the provided context.

Context ({scope} search):
{context}

Question: {question}

Provide a concise, accurate answer. If the context is insufficient, say so."""

    return call_mimo([
        {"role": "system", "content": "You answer questions based on graph data."},
        {"role": "user", "content": prompt},
    ], max_tokens=512)


def judge_answer(question: str, ground_truth: str, generated: str) -> Dict:
    """Judge answer quality using MiMo (1-5 scale)."""
    prompt = f"""Rate the quality of the generated answer on a scale of 1-5.

Question: {question}
Reference Answer: {ground_truth}
Generated Answer: {generated}

Scoring criteria:
- 5: Completely correct, all key information present
- 4: Mostly correct, minor omissions
- 3: Partially correct, some key info missing or slightly inaccurate
- 2: Mostly incorrect, only minor relevant info
- 1: Completely wrong or irrelevant

Respond with JSON only: {{"score": X, "reason": "brief explanation"}}"""

    response = call_mimo([
        {"role": "system", "content": "You are an answer quality judge. Output JSON only."},
        {"role": "user", "content": prompt},
    ], max_tokens=256)

    # Parse JSON from response
    import re
    json_match = re.search(r'\{[^}]+\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass

    # Fallback: extract score
    score_match = re.search(r'"score"\s*:\s*(\d)', response)
    score = int(score_match.group(1)) if score_match else 3
    return {"score": score, "reason": response[:100]}


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_algorithm(name: str, edge_file: Path, comm_file: Path,
                       qa_pairs: List[Dict]) -> Dict:
    """Evaluate one algorithm on all QA pairs."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    G = load_edge_list(edge_file)
    partition = load_partition(comm_file)

    scores = []
    details = []

    for i, qa in enumerate(qa_pairs):
        print(f"\n  QA {i+1}/{len(qa_pairs)}: {qa['question'][:60]}...")

        # Determine scope
        scope = qa.get("type", "local")

        # Retrieve context
        if scope == "global":
            context = simulate_global_search(qa["question"], G, partition, qa)
        else:
            context = simulate_local_search(qa["question"], G, partition, qa)

        # Generate answer
        answer = generate_answer(qa["question"], context, scope)
        print(f"    Generated: {answer[:80]}...")

        # Judge
        judgment = judge_answer(qa["question"], qa["ground_truth"], answer)
        score = judgment.get("score", 3)
        scores.append(score)
        details.append({
            "question": qa["question"],
            "ground_truth": qa["ground_truth"],
            "context": context[:200],
            "generated_answer": answer,
            "score": score,
            "reason": judgment.get("reason", ""),
        })
        print(f"    Score: {score}/5 | {judgment.get('reason', '')[:60]}...")

        # Rate limit protection
        time.sleep(1)

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\n  Average Score: {avg_score:.2f}/5")

    return {
        "algorithm": name,
        "num_qa": len(qa_pairs),
        "avg_score": round(avg_score, 3),
        "scores": scores,
        "details": details,
    }


def main():
    print("=" * 60)
    print("MiMo LLM End-to-End Community Detection Evaluation")
    print("=" * 60)

    # Use LFR small easy (fast, has ground truth communities)
    edge_file = DATA_DIR / "lfr_small_easy_edges.txt"
    louvain_file = DATA_DIR / "lfr_small_easy_partition_louvain.txt"
    leiden_file = DATA_DIR / "lfr_small_easy_partition_leiden.txt"

    if not edge_file.exists() or not louvain_file.exists() or not leiden_file.exists():
        print("ERROR: Required partition files not found.")
        print("Run evaluate_leiden.py first to generate partitions.")
        sys.exit(1)

    G = load_edge_list(edge_file)
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Generate QA pairs once (use Louvain partition as base)
    louvain_partition = load_partition(louvain_file)
    qa_pairs = generate_qa_pairs(G, louvain_partition, num_pairs=6)
    print(f"Generated {len(qa_pairs)} QA pairs")

    # Save QA pairs
    qa_file = BASE_DIR / "qa_pairs.json"
    with open(qa_file, "w") as f:
        json.dump(qa_pairs, f, indent=2)
    print(f"Saved QA pairs to {qa_file}")

    # Evaluate Louvain
    louvain_result = evaluate_algorithm("louvain", edge_file, louvain_file, qa_pairs)

    # Evaluate Leiden
    leiden_result = evaluate_algorithm("leiden", edge_file, leiden_file, qa_pairs)

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Louvain: {louvain_result['avg_score']:.2f}/5 ({louvain_result['num_qa']} QA pairs)")
    print(f"Leiden:  {leiden_result['avg_score']:.2f}/5 ({leiden_result['num_qa']} QA pairs)")
    delta = leiden_result['avg_score'] - louvain_result['avg_score']
    winner = "Leiden" if delta > 0 else "Louvain"
    print(f"Delta: {delta:+.2f} → {winner} wins")

    # Save results
    results = {
        "dataset": "lfr_small_easy",
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "qa_pairs": len(qa_pairs),
        "louvain": louvain_result,
        "leiden": leiden_result,
        "winner": winner,
        "delta": round(delta, 3),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
