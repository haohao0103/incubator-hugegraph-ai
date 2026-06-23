"""
GraphRAG Community Detection Quality Assessment
================================================

Four-layer evaluation framework for GraphRAG community detection.

Usage:
    python evaluate_graphrag_quality.py
"""

import json
import sys
import math
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

try:
    import networkx as nx
except ImportError:
    nx = None

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_FILE = BASE_DIR / "graphrag_quality_results.json"


@dataclass
class L2Metrics:
    avg_community_size: float = 0.0
    size_cv: float = 0.0
    size_gini: float = 0.0
    avg_internal_density: float = 0.0
    avg_external_density: float = 0.0
    density_ratio: float = 0.0
    avg_conductance: float = 0.0
    avg_cut_ratio: float = 0.0
    fragmentation_score: float = 0.0


@dataclass
class L3Metrics:
    entity_coverage: float = 0.0
    avg_entities_per_comm: float = 0.0
    relationship_density: float = 0.0
    cross_community_ratio: float = 0.0
    community_richness: float = 0.0
    isolated_node_ratio: float = 0.0
    avg_clustering_coeff: float = 0.0
    semantic_coherence: float = 0.0


@dataclass
class L4Metrics:
    simulated_recall: float = 0.0
    simulated_precision: float = 0.0
    information_coverage: float = 0.0
    granularity_score: float = 0.0
    query_answerability: float = 0.0


@dataclass
class GraphRAGQualityReport:
    dataset: str
    algorithm: str
    num_nodes: int
    num_edges: int
    num_communities: int
    modularity: float
    l2_structure: L2Metrics = field(default_factory=L2Metrics)
    l3_content: L3Metrics = field(default_factory=L3Metrics)
    l4_rag: L4Metrics = field(default_factory=L4Metrics)
    overall_score: float = 0.0
    grade: str = ""


def gini_coefficient(values: List[float]) -> float:
    if not values or len(values) < 2:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumsum = 0
    for i, v in enumerate(sorted_vals, 1):
        cumsum += (2 * i - n - 1) * v
    return cumsum / (n * sum(sorted_vals))


def coefficient_of_variation(values: List[float]) -> float:
    if not values or len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / mean


def load_edge_list(path: Path) -> "nx.Graph":
    G = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                G.add_edge(u, v)
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


def evaluate_l2_structure(G: "nx.Graph", partition: Dict[int, int]) -> L2Metrics:
    comms = communities_from_partition(partition)
    sizes = [len(members) for members in comms.values()]
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    
    if not sizes:
        return L2Metrics()
    
    avg_size = sum(sizes) / len(sizes)
    size_cv = coefficient_of_variation(sizes)
    size_gini = gini_coefficient(sizes)
    singletons = sum(1 for s in sizes if s == 1)
    fragmentation = singletons / len(sizes) if sizes else 0.0
    
    internal_densities, external_densities, conductances, cut_ratios = [], [], [], []
    
    for comm_id, members in comms.items():
        if len(members) <= 1:
            continue
        member_set = set(members)
        internal_edges = 0
        external_edges = 0
        for node in members:
            for neighbor in G.neighbors(node):
                if neighbor in member_set:
                    internal_edges += 1
                else:
                    external_edges += 1
        internal_edges = internal_edges // 2
        n = len(members)
        possible_internal = n * (n - 1) / 2
        possible_external = n * (total_nodes - n)
        
        int_density = internal_edges / possible_internal if possible_internal > 0 else 0
        internal_densities.append(int_density)
        ext_density = external_edges / possible_external if possible_external > 0 else 0
        external_densities.append(ext_density)
        
        vol_s = internal_edges * 2 + external_edges
        vol_complement = (total_edges * 2) - vol_s
        conductance = external_edges / min(vol_s, vol_complement) if min(vol_s, vol_complement) > 0 else 0
        conductances.append(conductance)
        
        cut_ratio = external_edges / (n * (total_nodes - n)) if n * (total_nodes - n) > 0 else 0
        cut_ratios.append(cut_ratio)
    
    avg_int = sum(internal_densities) / len(internal_densities) if internal_densities else 0
    avg_ext = sum(external_densities) / len(external_densities) if external_densities else 0
    density_ratio = avg_int / avg_ext if avg_ext > 0 else 0
    
    return L2Metrics(
        avg_community_size=avg_size,
        size_cv=size_cv,
        size_gini=size_gini,
        avg_internal_density=avg_int,
        avg_external_density=avg_ext,
        density_ratio=density_ratio,
        avg_conductance=sum(conductances) / len(conductances) if conductances else 0,
        avg_cut_ratio=sum(cut_ratios) / len(cut_ratios) if cut_ratios else 0,
        fragmentation_score=fragmentation,
    )


def evaluate_l3_content(G: "nx.Graph", partition: Dict[int, int]) -> L3Metrics:
    comms = communities_from_partition(partition)
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    
    if not comms:
        return L3Metrics()
    
    non_singleton_nodes = sum(len(m) for m in comms.values() if len(m) > 1)
    entity_coverage = non_singleton_nodes / total_nodes if total_nodes > 0 else 0
    avg_entities = sum(len(m) for m in comms.values()) / len(comms)
    
    cross_edges = sum(1 for u, v in G.edges() if partition.get(u) != partition.get(v))
    cross_ratio = cross_edges / total_edges if total_edges > 0 else 0
    
    isolated = 0
    for members in comms.values():
        if len(members) <= 1:
            isolated += 1
            continue
        member_set = set(members)
        has_internal = False
        for node in members:
            for neighbor in G.neighbors(node):
                if neighbor in member_set:
                    has_internal = True
                    break
            if has_internal:
                break
        if not has_internal:
            isolated += len(members)
    isolated_ratio = isolated / total_nodes if total_nodes > 0 else 0
    
    clustering_scores = []
    for members in comms.values():
        if len(members) < 3:
            continue
        subgraph = G.subgraph(members)
        try:
            clustering_scores.append(nx.average_clustering(subgraph))
        except:
            pass
    avg_clustering = sum(clustering_scores) / len(clustering_scores) if clustering_scores else 0
    
    richness = (
        (1 - gini_coefficient([len(m) for m in comms.values()])) * 0.3 +
        (1 - isolated_ratio) * 0.4 +
        min(avg_entities / 50, 1.0) * 0.3
    )
    
    return L3Metrics(
        entity_coverage=entity_coverage,
        avg_entities_per_comm=avg_entities,
        relationship_density=avg_clustering,
        cross_community_ratio=cross_ratio,
        community_richness=richness,
        isolated_node_ratio=isolated_ratio,
        avg_clustering_coeff=avg_clustering,
        semantic_coherence=1 - cross_ratio,
    )


def evaluate_l4_rag(G: "nx.Graph", partition: Dict[int, int], num_communities: int) -> L4Metrics:
    comms = communities_from_partition(partition)
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    
    if not comms:
        return L4Metrics()
    
    sizes = [len(m) for m in comms.values()]
    avg_size = sum(sizes) / len(sizes)
    
    if avg_size < 5:
        recall = 0.3
    elif avg_size < 20:
        recall = 0.6
    elif avg_size < 100:
        recall = 0.9
    elif avg_size < 500:
        recall = 0.75
    else:
        recall = 0.5
    
    cross_edges = sum(1 for u, v in G.edges() if partition.get(u) != partition.get(v))
    cross_ratio = cross_edges / total_edges if total_edges > 0 else 0
    precision = max(0.1, min(1.0, 1.0 - cross_ratio * 2))
    
    sorted_comms = sorted(comms.values(), key=len, reverse=True)
    top_nodes = sum(len(c) for c in sorted_comms[:10])
    info_coverage = top_nodes / total_nodes if total_nodes > 0 else 0
    
    num_comms = len(comms)
    ideal_comms = max(10, total_nodes // 50)
    granularity = max(0.0, min(1.0, 1.0 - abs(num_comms - ideal_comms) / max(num_comms, ideal_comms)))
    
    answerability = recall * 0.4 + precision * 0.3 + info_coverage * 0.2 + granularity * 0.1
    
    return L4Metrics(
        simulated_recall=recall,
        simulated_precision=precision,
        information_coverage=info_coverage,
        granularity_score=granularity,
        query_answerability=answerability,
    )


def calculate_overall_score(report: GraphRAGQualityReport) -> Tuple[float, str]:
    l2 = report.l2_structure
    l3 = report.l3_content
    l4 = report.l4_rag
    
    scores = {
        "modularity": max(0, report.modularity) * 0.10,
        "density_ratio": min(l2.density_ratio, 5.0) / 5.0 * 0.05,
        "conductance": (1.0 - l2.avg_conductance) * 0.05,
        "fragmentation": (1.0 - l2.fragmentation_score) * 0.05,
        "size_balance": (1.0 - min(l2.size_gini, 1.0)) * 0.05,
        "entity_coverage": l3.entity_coverage * 0.10,
        "richness": l3.community_richness * 0.10,
        "clustering": l3.avg_clustering_coeff * 0.05,
        "coherence": l3.semantic_coherence * 0.05,
        "isolation": (1.0 - l3.isolated_node_ratio) * 0.05,
        "recall": l4.simulated_recall * 0.10,
        "precision": l4.simulated_precision * 0.10,
        "info_coverage": l4.information_coverage * 0.05,
        "granularity": l4.granularity_score * 0.05,
        "answerability": l4.query_answerability * 0.05,
    }
    
    total = sum(scores.values())
    
    if total >= 0.85:
        grade = "A+"
    elif total >= 0.75:
        grade = "A"
    elif total >= 0.65:
        grade = "B"
    elif total >= 0.50:
        grade = "C"
    elif total >= 0.35:
        grade = "D"
    else:
        grade = "F"
    
    return total, grade


def run_community_detection(G: "nx.Graph", algorithm: str) -> Tuple[Dict[int, int], float]:
    start = time.time()
    
    if algorithm == "louvain":
        communities = nx.community.louvain_communities(G, seed=42)
        partition = {}
        for comm_id, comm in enumerate(communities):
            for node in comm:
                partition[node] = comm_id
    elif algorithm == "leiden":
        try:
            import igraph as ig
            import leidenalg
            ig_G = ig.Graph.TupleList(G.edges(), directed=False)
            nx_nodes = list(G.nodes())
            partition_leiden = leidenalg.find_partition(ig_G, leidenalg.ModularityVertexPartition)
            partition = {}
            for ig_idx, comm_id in enumerate(partition_leiden.membership):
                partition[nx_nodes[ig_idx]] = comm_id
        except ImportError:
            print("    leidenalg not available, falling back to louvain")
            return run_community_detection(G, "louvain")
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    comm_sets = defaultdict(set)
    for node, comm in partition.items():
        comm_sets[comm].add(node)
    Q = nx.community.modularity(G, [s for s in comm_sets.values()])
    
    return partition, Q


def save_partition(partition: Dict[int, int], path: Path):
    with open(path, "w") as f:
        for node, comm in sorted(partition.items()):
            f.write(f"{node} {comm}\n")


def evaluate_dataset(name: str, edge_file: Path, comm_file: Path, algorithm: str, modularity: float) -> GraphRAGQualityReport:
    print(f"\n  [{algorithm.upper()}]")
    
    G = load_edge_list(edge_file)
    partition = load_partition(comm_file)
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_comms = len(set(partition.values()))
    
    print(f"    Nodes: {num_nodes:,}, Edges: {num_edges:,}, Communities: {num_comms}")
    
    l2 = evaluate_l2_structure(G, partition)
    l3 = evaluate_l3_content(G, partition)
    l4 = evaluate_l4_rag(G, partition, num_comms)
    
    report = GraphRAGQualityReport(
        dataset=name,
        algorithm=algorithm,
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_communities=num_comms,
        modularity=modularity,
        l2_structure=l2,
        l3_content=l3,
        l4_rag=l4,
    )
    
    report.overall_score, report.grade = calculate_overall_score(report)
    
    return report


def main():
    if nx is None:
        print("ERROR: networkx is required.")
        sys.exit(1)
    
    all_results: List[Dict] = []
    
    datasets = []
    for name in ["lfr_small_easy", "lfr_small_medium", "lfr_small_hard",
                 "lfr_medium_easy", "lfr_medium_medium"]:
        edge_file = DATA_DIR / f"{name}_edges.txt"
        if edge_file.exists():
            datasets.append((name, edge_file))
    
    for name in ["amazon", "dblp", "youtube"]:
        edge_file = DATA_DIR / f"{name}.txt"
        if edge_file.exists():
            datasets.append((name, edge_file))
    
    if not datasets:
        print("No datasets found! Run download_datasets.py first.")
        sys.exit(1)
    
    print(f"\nFound {len(datasets)} datasets. Running Louvain + Leiden for each...")
    
    for name, edge_file in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")
        
        G = load_edge_list(edge_file)
        print(f"  Loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        
        for algo in ["louvain", "leiden"]:
            comm_file = DATA_DIR / f"{name}_partition_{algo}.txt"
            
            if comm_file.exists():
                print(f"  → Loading existing {algo} partition...")
                partition = load_partition(comm_file)
                comm_sets = defaultdict(set)
                for node, comm in partition.items():
                    comm_sets[comm].add(node)
                mod = nx.community.modularity(G, [s for s in comm_sets.values()])
            else:
                print(f"  → Running {algo} community detection...")
                partition, mod = run_community_detection(G, algo)
                save_partition(partition, comm_file)
                print(f"    Saved partition to {comm_file.name}")
            
            report = evaluate_dataset(name, edge_file, comm_file, algo, mod)
            all_results.append(asdict(report))
            print(f"  → Score: {report.overall_score:.3f} | Grade: {report.grade}")
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {RESULTS_FILE}")
    
    # Summary
    print("\n" + "="*90)
    print("GRAPHRAQ COMMUNITY QUALITY: Leiden vs Louvain")
    print("="*90)
    
    by_dataset = defaultdict(dict)
    for r in all_results:
        by_dataset[r["dataset"]][r["algorithm"]] = r
    
    print(f"{'Dataset':<20} {'Metric':<25} {'Louvain':>10} {'Leiden':>10} {'Δ':>8} {'Winner':>8}")
    print("-"*90)
    
    for ds in sorted(by_dataset.keys()):
        louvain = by_dataset[ds].get("louvain")
        leiden_data = by_dataset[ds].get("leiden")
        if not louvain or not leiden_data:
            continue
        
        winner = "Leiden" if leiden_data['overall_score'] > louvain['overall_score'] else "Louvain"
        
        print(f"{ds:<20} {'Overall Score':<25} {louvain['overall_score']:>10.3f} {leiden_data['overall_score']:>10.3f} {leiden_data['overall_score']-louvain['overall_score']:>+8.3f} {winner:>8}")
        print(f"{'':<20} {'Grade':<25} {louvain['grade']:>10} {leiden_data['grade']:>10} {'':>8} {'':>8}")
        print(f"{'':<20} {'L2 Density Ratio':<25} {louvain['l2_structure']['density_ratio']:>10.3f} {leiden_data['l2_structure']['density_ratio']:>10.3f} {leiden_data['l2_structure']['density_ratio']-louvain['l2_structure']['density_ratio']:>+8.3f} {'':>8}")
        print(f"{'':<20} {'L2 Conductance':<25} {louvain['l2_structure']['avg_conductance']:>10.3f} {leiden_data['l2_structure']['avg_conductance']:>10.3f} {leiden_data['l2_structure']['avg_conductance']-louvain['l2_structure']['avg_conductance']:>+8.3f} {'':>8}")
        print(f"{'':<20} {'L3 Entity Coverage':<25} {louvain['l3_content']['entity_coverage']:>10.3f} {leiden_data['l3_content']['entity_coverage']:>10.3f} {leiden_data['l3_content']['entity_coverage']-louvain['l3_content']['entity_coverage']:>+8.3f} {'':>8}")
        print(f"{'':<20} {'L3 Richness':<25} {louvain['l3_content']['community_richness']:>10.3f} {leiden_data['l3_content']['community_richness']:>10.3f} {leiden_data['l3_content']['community_richness']-louvain['l3_content']['community_richness']:>+8.3f} {'':>8}")
        print(f"{'':<20} {'L4 Answerability':<25} {louvain['l4_rag']['query_answerability']:>10.3f} {leiden_data['l4_rag']['query_answerability']:>10.3f} {leiden_data['l4_rag']['query_answerability']-louvain['l4_rag']['query_answerability']:>+8.3f} {'':>8}")
        print()


if __name__ == "__main__":
    main()
