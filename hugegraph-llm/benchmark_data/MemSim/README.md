# MemSim / MemDaily Dataset

## Source

- Repository: https://github.com/nuster1128/MemSim
- Paper: Zhang et al., "MemSim: A Bayesian Simulator for Evaluating Memory of LLM-based Personal Assistants", arXiv:2409.20163, 2024.
- File: `data_generation/final_dataset/memdaily.json`
- Downloaded: 2026-06-26 from GitHub raw

## Dataset Description

MemDaily is a Chinese daily-life memory simulation dataset for evaluating memory mechanisms in LLM-based personal assistants.

| Split | Trajectories | QA Type |
|-------|-------------|---------|
| simple | 500 | direct recall |
| conditional | 500 | conditional recall |
| comparative | 492 | comparative reasoning |
| aggregative | 462 | aggregation |
| post_processing | 500 | post-processed recall |
| noisy | 500 | noise-resilient recall |
| **Total** | **2,954** | |

## License

Please refer to the original MemSim repository for the exact license. This dataset is redistributed here solely for reproducibility of the HugeGraph-AI memory benchmark.

## Usage

The benchmark script `src/hugegraph_llm/poc/benchmark_memsim.py` reads this file by default. You can override the path via environment variable:

```bash
export MEMSIM_DATA_PATH=/path/to/memdaily.json
```
