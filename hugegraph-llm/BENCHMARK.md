# HugeGraph-AI Benchmark Suite

This document describes how to run the GraphRAG-Bench and MemSim benchmarks included in this branch.

All benchmarks read LLM / HugeGraph configuration from `hugegraph-llm/.env`. Copy `.env.example` to `.env` and fill in your API keys first:

```bash
cp hugegraph-llm/.env.example hugegraph-llm/.env
# edit hugegraph-llm/.env
```

---

## 1. GraphRAG-Bench P0-Improved

### What it evaluates

End-to-end GraphRAG pipeline on the GraphRAG-Bench dataset (ICLR'26):

- **Real embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Real retrieval**: FAISS vector + BM25 full-text + HugeGraph traversal
- **Real LLM**: any OpenAI-compatible chat endpoint (default MiMo v2.5 Pro)
- **Real graph DB**: HugeGraph REST API
- **RRF fusion** of the three retrieval channels

### Dataset

Located at `hugegraph-llm/benchmark_data/GraphRAG-Bench/`. Included in this branch.

### Run

```bash
cd hugegraph-llm
python src/hugegraph_llm/poc/graphrag_bench_p0_improved.py
```

### Configuration (via `.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_CHAT_API_BASE` | `https://api.xiaomimimo.com/v1` | Chat completions base URL |
| `OPENAI_CHAT_API_KEY` | *required* | API key |
| `OPENAI_CHAT_LANGUAGE_MODEL` | `mimo-v2.5-pro` | Model name |
| `GRAPH_URL` | `http://127.0.0.1:8080` | HugeGraph REST URL |
| `GRAPH_NAME` | `hugegraph` | Graph name |
| `GRAPH_USER` | `admin` | HugeGraph user |
| `GRAPH_PWD` | `xxx` | HugeGraph password |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `EMBEDDING_DIM` | `384` | Embedding dimension |

### Output

- `hugegraph-llm/poc_results/graphrag_bench_p0_improved_result.json`
- Console summary with F1, EM, Latency per domain.

### Runtime

~15-30 minutes for 120 questions (30 per domain × 2 domains + entity extraction), depending on LLM latency.

---

## 2. MemSim Chinese Memory Benchmark

### What it evaluates

Memory retrieval and QA on the MemDaily Chinese daily-life dataset. The benchmark:

- Builds a per-trajectory vector index with `all-MiniLM-L6-v2`
- Runs a custom BM25 over word + character tokens
- Fuses vector and BM25 rankings with RRF (k=40)
- Uses an LLM to answer 4-choice MCQ questions
- Reports exact-match accuracy per split

### Dataset

Located at `hugegraph-llm/benchmark_data/MemSim/memdaily.json`. Downloaded from https://github.com/nuster1128/MemSim on 2026-06-26.

### Run

```bash
cd hugegraph-llm
python src/hugegraph_llm/poc/benchmark_memsim.py
```

### Configuration (via `.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_CHAT_API_BASE` | `https://api.xiaomimimo.com/v1` | Chat completions base URL |
| `OPENAI_CHAT_API_KEY` | *required* | API key |
| `OPENAI_CHAT_LANGUAGE_MODEL` | `mimo-v2.5-pro` | Model name |
| `MEMSIM_DATA_PATH` | `benchmark_data/MemSim/memdaily.json` | Dataset path |
| `MEMSIM_SAMPLE_PER_SPLIT` | `20` | Trajectories sampled per split |

### Output

- `hugegraph-llm/src/hugegraph_llm/poc/benchmark_memsim_result.json`
- Console accuracy table per split and overall.

### Runtime

~5-10 minutes for 120 sampled questions, depending on LLM latency.

---

## 3. CodeGraph Benchmark

See `../benchmark/README.md` and `../REPRODUCE.md` section 8.
