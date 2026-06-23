#!/usr/bin/env python3
"""Download standard community detection benchmark datasets."""

import os
import urllib.request
import gzip
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# Dataset URLs from SNAP Stanford
DATASETS = {
    "amazon": {
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz",
        "ground_truth": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.top5000.cmty.txt.gz",
        "desc": "Amazon product co-purchasing network (334K nodes, 925K edges)",
    },
    "dblp": {
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
        "ground_truth": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.all.cmty.txt.gz",
        "desc": "DBLP collaboration network (317K nodes, 1M edges)",
    },
    "youtube": {
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",
        "ground_truth": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.all.cmty.txt.gz",
        "desc": "YouTube social network (1.1M nodes, 3M edges)",
    },
    "livejournal": {
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz",
        "ground_truth": "https://snap.stanford.edu/data/bigdata/communities/com-lj.all.cmty.txt.gz",
        "desc": "LiveJournal social network (4M nodes, 34M edges)",
    },
}

# LFR benchmark parameters for synthetic data
LFR_CONFIGS = [
    {"n": 1000, "tau1": 3, "tau2": 1.5, "mu": 0.1, "min_community": 20, "name": "lfr_small_easy"},
    {"n": 1000, "tau1": 3, "tau2": 1.5, "mu": 0.3, "min_community": 20, "name": "lfr_small_medium"},
    {"n": 1000, "tau1": 3, "tau2": 1.5, "mu": 0.5, "min_community": 20, "name": "lfr_small_hard"},
    {"n": 10000, "tau1": 3, "tau2": 1.5, "mu": 0.1, "min_community": 20, "name": "lfr_medium_easy"},
    {"n": 10000, "tau1": 3, "tau2": 1.5, "mu": 0.3, "min_community": 20, "name": "lfr_medium_medium"},
    {"n": 10000, "tau1": 3, "tau2": 1.5, "mu": 0.5, "min_community": 20, "name": "lfr_medium_hard"},
]


def download_file(url: str, dest: Path):
    """Download a file with progress."""
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return
    print(f"  Downloading {dest.name}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Done: {dest.name}")


def gunzip_file(src: Path, dest: Path):
    """Decompress a gzip file."""
    if dest.exists():
        return
    print(f"  Decompressing {src.name}...")
    with gzip.open(src, "rb") as f_in:
        with open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"  Decompressed: {dest.name}")


def download_snap_datasets():
    """Download SNAP community detection datasets."""
    print("=" * 60)
    print("Downloading SNAP Datasets")
    print("=" * 60)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, info in DATASETS.items():
        print(f"\n[{name}] {info['desc']}")
        # Download graph
        graph_gz = DATA_DIR / f"{name}.txt.gz"
        download_file(info["url"], graph_gz)
        graph_txt = DATA_DIR / f"{name}.txt"
        gunzip_file(graph_gz, graph_txt)

        # Download ground truth
        gt_gz = DATA_DIR / f"{name}_gt.txt.gz"
        download_file(info["ground_truth"], gt_gz)
        gt_txt = DATA_DIR / f"{name}_gt.txt"
        gunzip_file(gt_gz, gt_txt)


def generate_lfr_benchmarks():
    """Generate LFR benchmark synthetic graphs with ground truth."""
    print("\n" + "=" * 60)
    print("Generating LFR Benchmark Synthetic Graphs")
    print("=" * 60)

    try:
        import networkx as nx
        import numpy as np
    except ImportError:
        print("ERROR: networkx or numpy not installed")
        return

    for cfg in LFR_CONFIGS:
        name = cfg["name"]
        edge_file = DATA_DIR / f"{name}_edges.txt"
        comm_file = DATA_DIR / f"{name}_communities.txt"

        if edge_file.exists() and comm_file.exists():
            print(f"  Already exists: {name}")
            continue

        print(f"\n  Generating {name} (n={cfg['n']}, mu={cfg['mu']})...")

        try:
            # Use networkx's LFR generator if available
            G = nx.LFR_benchmark_graph(
                n=cfg["n"],
                tau1=cfg["tau1"],
                tau2=cfg["tau2"],
                mu=cfg["mu"],
                min_community=cfg["min_community"],
                seed=42,
            )
        except Exception as e:
            print(f"  networkx LFR failed: {e}, using fallback...")
            G = _generate_lfr_fallback(cfg)

        # Save edges
        with open(edge_file, "w") as f:
            f.write("# Source Target\n")
            for u, v in G.edges():
                f.write(f"{u} {v}\n")

        # Save ground truth communities
        communities = {}
        for node in G.nodes():
            if hasattr(G.nodes[node], "community"):
                comm = G.nodes[node]["community"]
            elif "community" in G.nodes[node]:
                comm = G.nodes[node]["community"]
            else:
                # Fallback: each node in its own community
                comm = {node}
            for c in (comm if isinstance(comm, (set, frozenset, list, tuple)) else [comm]):
                communities.setdefault(c, set()).add(node)

        with open(comm_file, "w") as f:
            f.write("# Community nodes (one community per line)\n")
            for comm_id, nodes in communities.items():
                f.write(" ".join(map(str, sorted(nodes))) + "\n")

        print(f"  Saved: {len(G.nodes)} nodes, {len(G.edges)} edges, {len(communities)} communities")


def _generate_lfr_fallback(cfg: dict) -> "nx.Graph":
    """Simple fallback graph generator when LFR fails."""
    import networkx as nx
    import random

    random.seed(42)
    G = nx.Graph()
    n = cfg["n"]
    G.add_nodes_from(range(n))

    # Assign communities
    comm_size = cfg["min_community"]
    num_communities = n // comm_size
    communities = {}
    for i in range(n):
        c = i // comm_size
        if c >= num_communities:
            c = num_communities - 1
        communities[i] = c
        G.nodes[i]["community"] = {c}

    # Add edges: mix of intra-community and inter-community
    mu = cfg["mu"]
    avg_degree = 6
    for i in range(n):
        for _ in range(avg_degree // 2):
            if random.random() > mu:
                # Intra-community edge
                c = communities[i]
                candidates = [j for j in range(n) if communities[j] == c and j != i]
                if candidates:
                    j = random.choice(candidates)
                    G.add_edge(i, j)
            else:
                # Inter-community edge
                c = communities[i]
                candidates = [j for j in range(n) if communities[j] != c]
                if candidates:
                    j = random.choice(candidates)
                    G.add_edge(i, j)

    return G


def main():
    print("Community Detection Benchmark Dataset Downloader")
    print("=" * 60)

    # Generate LFR benchmarks first (always available)
    generate_lfr_benchmarks()

    # Try to download SNAP datasets (may fail due to size)
    print("\n" + "=" * 60)
    print("SNAP Datasets (large, may take time)")
    print("=" * 60)
    try:
        download_snap_datasets()
    except Exception as e:
        print(f"\nWarning: Some SNAP datasets failed to download: {e}")
        print("LFR benchmarks are sufficient for validation.")

    print("\n" + "=" * 60)
    print("Dataset download complete!")
    print(f"Data directory: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
