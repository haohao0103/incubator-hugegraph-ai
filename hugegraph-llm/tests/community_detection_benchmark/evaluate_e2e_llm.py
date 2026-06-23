"""
End-to-End LLM Evaluation for GraphRAG Community Detection
============================================================

Uses MiMo LLM to evaluate Leiden vs Louvain on real QA tasks.

Pipeline:
1. Load graph + community partitions (Leiden & Louvain)
2. Generate community summaries with LLM
3. Generate QA pairs from community structure
4. For each QA pair:
   a. Retrieve relevant communities (simulated)
   b. Generate answer with MiMo using community summaries
   c. Judge answer quality with MiMo
5. Compare average scores: Leiden vs Louvain

Usage:
    python evaluate_e2e_llm.py
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# MiMo API client
from openai import OpenAI

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_FILE = BASE_DIR / "e2e_llm_results.json"

LLM_CLIENT = OpenAI(
    base_url="https://api.xiaomimimo.com/v1",
    api_key="sk-cs5kqi80f6upqy2e3k3xi39jtizhpgf6dkdd3j9ysoupfw7p",
)
LLM_MODEL = "mimo-v2.5-pro"


def llm_call(messages: List[Dict], max_tokens: int = 512) -> str:
    """Call MiMo LLM and extract text from reasoning model."""
    try:
        response = LLM_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=max_tokens,
        )
        msg = response.choices[0].message
        # For reasoning models, content may be empty, check reasoning_content
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", "") or ""
        return (content + reasoning).strip()
    except Exception as e:
        print(f"[LLM Error] {e}")
        return ""


def load_graph_and_partitions(name: str) -> Tuple[nx.Graph, Dict[int, int], Dict[int, int]]:
    """Load graph and both Leiden/Louvain partitions."""
    edge_file = DATA_DIR / f"{name}_edges.txt"
    G = nx.Graph()
    with open(edge_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                G.add_edge(int(parts[0]), int(parts[1]))

    partitions = {}
    for algo in ["louvain", "leiden"]:
        comm_file = DATA_DIR / f"{name}_partition_{algo}.txt"
        partition = {}
        with open(comm_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    partition[int(parts[0])] = int(parts[1])
        partitions[algo] = partition

    return G, partitions["louvain"], partitions["leiden"]


def get_communities(partition: Dict[int, int]) -> Dict[int, List[int]]:
    """Convert node->comm to comm->[nodes]."""
    comms = defaultdict(list)
    for node, comm in partition.items():
        comms[comm].append(node)
    return dict(comms)


def generate_community_summary(G: nx.Graph, members: List[int], comm_id: int) -> str:
    """Generate a summary for a community using LLM."""
    subgraph = G.subgraph(members)
    num_nodes = len(members)
    num_edges = subgraph.number_of_edges()
    density = nx.density(subgraph) if num_nodes > 1 else 0

    # Find top connected nodes (hub entities)
    degrees = {n: subgraph.degree(n) for n in members}
    top_nodes = sorted(degrees.items(), key=lambda x: -x[1])[:5]
    top_nodes_str = ", ".join([f"Node-{n} (deg={d})" for n, d in top_nodes])

    prompt = f"""You are analyzing a community in a knowledge graph.

Community ID: {comm_id}
Size: {num_nodes} nodes, {num_edges} edges
Density: {density:.3f}
Top connected entities: {top_nodes_str}

Write a concise 2-3 sentence summary describing what this community represents.
Focus on themes, relationships, and key entities."""

    summary = llm_call([
        {"role": "system", "content": "You summarize graph communities concisely."},
        {"role": "user", "content": prompt},
    ], max_tokens=256)

    return summary or f"Community {comm_id} with {num_nodes} nodes and {num_edges} edges."


def generate_qa_pairs(G: nx.Graph, partition: Dict[int, int], num_pairs: int = 10) -> List[Dict]:
    """Generate QA pairs from graph structure."""
    comms = get_communities(partition)
    qa_pairs = []

    # Type 1: Entity existence (local)
    nodes = list(G.nodes())
    for i in range(min(num_pairs // 3, len(nodes))):
        node = nodes[i]
        neighbors = list(G.neighbors(node))
        if neighbors:
            qa_pairs.append({
                "question": f"What is connected to Node-{node}?",
                "answer": f"Node-{node} is connected to: " + ", ".join([f"Node-{n}" for n in neighbors[:5]]),
                "type": "local",
                "relevant_nodes": [node] + neighbors[:5],
            })

    # Type 2: Community membership (global-ish)
    for comm_id, members in list(comms.items())[:num_pairs // 3]:
        if len(members) >= 3:
            qa_pairs.append({
                "question": f"What entities are in Community {comm_id}?",
                "answer": f"Community {comm_id} contains: " + ", ".join([f"Node-{n}" for n in members[:10]]),
                "type": "global",
                "relevant_community": comm_id,
            })

    # Type 3: Path/relationship questions (local)
    edges = list(G.edges())
    for i in range(min(num_pairs // 3, len(edges))):
        u, v = edges[i]
        qa_pairs.append({
            "question": f"What is the relationship between Node-{u} and Node-{v}?",
            "answer": f"Node-{u} and Node-{v} are directly connected by an edge.",
            "type": "local",
            "relevant_nodes": [u, v],
        })

    return qa_pairs[:num_pairs]


def retrieve_for_question(
    question: str,
    G: nx.Graph,
    partition: Dict[int, int],
    summaries: Dict[int, str],
    top_k: int = 3,
) -> List[int]:
    """Simulate community retrieval for a question.

    Simple heuristic: extract node IDs from question, find their communities.
    """
    import re
    node_ids = [int(x) for x in re.findall(r"Node-(\d+)", question)]
    comms = get_communities(partition)

    # Score communities by how many query nodes they contain
    comm_scores = defaultdict(int)
    for nid in node_ids:
        comm = partition.get(nid)
        if comm is not None:
            comm_scores[comm] += 1

    # Also score by keyword overlap with summaries
    question_lower = question.lower()
    for comm_id, summary in summaries.items():
        words = set(re.findall(r"[a-z]+", summary.lower()))
        qwords = set(re.findall(r"[a-z]+", question_lower))
        overlap = len(words & qwords)
        comm_scores[comm_id] += overlap * 0.5

    top_comms = sorted(comm_scores.items(), key=lambda x: -x[1])[:top_k]
    return [c[0] for c in top_comms]


def generate_answer(question: str, retrieved_summaries: List[str]) -> str:
    """Use MiMo to generate answer from retrieved community summaries."""
    context = "\n\n".join([f"Community Summary {i+1}:\n{s}" for i, s in enumerate(retrieved_summaries)])

    prompt = f"""Answer the question using the provided community summaries.

Community Summaries:
{context}

Question: {question}

Provide a concise answer based on the information above."""

    return llm_call([
        {"role": "system", "content": "You answer questions based on community summaries."},
        {"role": "user", "content": prompt},
    ], max_tokens=256)


def judge_answer(question: str, expected: str, generated: str) -> Tuple[int, str]:
    """Use MiMo as judge to score answer quality (1-5)."""
    prompt = f"""Rate the quality of the generated answer compared to the expected answer.

Question: {question}
Expected Answer: {expected}
Generated Answer: {generated}

Score (1-5):
- 5: Correct and complete, covers all key information
- 4: Mostly correct with minor omissions
- 3: Partially correct, some key info missing
- 2: Mostly incorrect or irrelevant
- 1: Completely wrong or unrelated

Respond with ONLY a JSON object: {{"score": X, "reason": "brief explanation"}}"""

    response = llm_call([
        {"role": "system", "content": "You are a strict answer quality judge. Output only JSON."},
        {"role": "user", "content": prompt},
    ], max_tokens=128)

    # Parse JSON from response
    try:
        import json as json_mod
        # Find JSON in response
        start = response.find("{")
        end = response.rfind("}")
        if start >= 0 and end > start:
            obj = json_mod.loads(response[start:end+1])
            return int(obj.get("score", 3)), obj.get("reason", "")
    except Exception:
        pass

    # Fallback: simple keyword overlap
    expected_words = set(expected.lower().split())
    generated_words = set(generated.lower().split())
    overlap = len(expected_words & generated_words)
    total = len(expected_words)
    ratio = overlap / total if total > 0 else 0
    if ratio > 0.7:
        return 4, "High keyword overlap"
    elif ratio > 0.4:
        return 3, "Moderate keyword overlap"
    elif ratio > 0.1:
        return 2, "Low keyword overlap"
    else:
        return 1, "Minimal keyword overlap"


def evaluate_algorithm(
    G: nx.Graph,
    partition: Dict[int, int],
    qa_pairs: List[Dict],
    algo_name: str,
) -> Dict:
    """Evaluate one algorithm on all QA pairs."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {algo_name}")
    print(f"{'='*60}")

    comms = get_communities(partition)
    print(f"Communities: {len(comms)}")

    # Generate community summaries
    print("Generating community summaries with LLM...")
    summaries = {}
    for comm_id in sorted(comms.keys())[:20]:  # Limit to 20 for speed
        summaries[comm_id] = generate_community_summary(G, comms[comm_id], comm_id)
        print(f"  Comm {comm_id}: {summaries[comm_id][:80]}...")

    # Evaluate each QA pair
    scores = []
    for i, qa in enumerate(qa_pairs):
        print(f"\n  QA {i+1}/{len(qa_pairs)}: {qa['question']}")

        # Retrieve relevant communities
        retrieved = retrieve_for_question(qa["question"], G, partition, summaries, top_k=3)
        retrieved_summaries = [summaries.get(c, "") for c in retrieved if c in summaries]

        print(f"    Retrieved communities: {retrieved}")

        # Generate answer
        answer = generate_answer(qa["question"], retrieved_summaries)
        print(f"    Generated: {answer[:100]}...")

        # Judge
        score, reason = judge_answer(qa["question"], qa["answer"], answer)
        scores.append(score)
        print(f"    Score: {score}/5 - {reason}")

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\n  Average Score: {avg_score:.2f}/5")

    return {
        "algorithm": algo_name,
        "num_communities": len(comms),
        "avg_score": avg_score,
        "scores": scores,
        "summaries": {str(k): v for k, v in summaries.items()},
    }


def main():
    dataset = "lfr_small_easy"
    print(f"Loading {dataset}...")
    G, louvain_partition, leiden_partition = load_graph_and_partitions(dataset)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Generate QA pairs (use Louvain partition as ground truth reference)
    print("\nGenerating QA pairs...")
    qa_pairs = generate_qa_pairs(G, louvain_partition, num_pairs=6)
    for qa in qa_pairs:
        print(f"  Q: {qa['question']}")
        print(f"  A: {qa['answer'][:80]}...")

    # Evaluate both algorithms
    louvain_result = evaluate_algorithm(G, louvain_partition, qa_pairs, "louvain")
    leiden_result = evaluate_algorithm(G, leiden_partition, qa_pairs, "leiden")

    # Save results
    results = {
        "dataset": dataset,
        "qa_pairs": qa_pairs,
        "louvain": louvain_result,
        "leiden": leiden_result,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Algorithm':<15} {'Communities':>12} {'Avg Score':>12}")
    print(f"{'-'*40}")
    print(f"{'Louvain':<15} {louvain_result['num_communities']:>12} {louvain_result['avg_score']:>11.2f}/5")
    print(f"{'Leiden':<15} {leiden_result['num_communities']:>12} {leiden_result['avg_score']:>11.2f}/5")
    delta = leiden_result['avg_score'] - louvain_result['avg_score']
    winner = "Leiden" if delta > 0 else "Louvain"
    print(f"\nDelta: {delta:+.2f} -> {winner} wins")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
