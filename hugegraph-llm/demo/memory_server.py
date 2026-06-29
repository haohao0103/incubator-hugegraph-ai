"""
HugeGraph Memory Demo Server
=============================
Thin wrapper around the production-grade memory_backend.py.

- Real graph storage: HugeGraph 1.7.0 (via PyHugeClient)
- Real vector index: FAISS
- Real fulltext index: BM25
- LLM: MiMo v2.5-pro (entity extraction / ranking / answer generation)
- Serves interactive HTML frontend from ../src/hugegraph_llm/poc/

Usage:
    export LLM_API_KEY=sk-...
    python demo/memory_server.py [--port 8765]
"""

import os
import sys
import argparse

# Load the workspace .env file before any hugegraph_llm imports so that
# OPENAI_CHAT_API_KEY / LLM_API_KEY are available in os.environ.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, "..", ".env")
if os.path.exists(ENV_PATH):
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH, override=False)
    except Exception:
        pass

# Ensure LLM_API_KEY is populated before importing memory_backend, which reads
# it at module import time.
if not os.environ.get("LLM_API_KEY"):
    os.environ["LLM_API_KEY"] = os.environ.get("OPENAI_CHAT_API_KEY", "")

# Make the current workspace's source tree take precedence over any
# venv-installed hugegraph-llm package.
WORKSPACE_SRC = os.path.join(SCRIPT_DIR, "..", "src")
if WORKSPACE_SRC not in sys.path:
    sys.path.insert(0, WORKSPACE_SRC)

from flask import Flask, send_from_directory
from hugegraph_llm.poc.memory_backend import create_app, MemoryPipelineBackend, HugeGraphMemoryClient


def main():
    parser = argparse.ArgumentParser(description="HugeGraph Memory Demo Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--graph-name", type=str, default=os.environ.get("HUGEGRAPH_GRAPH", "hugegraph"),
                        help="HugeGraph graph name to use")
    args = parser.parse_args()

    llm_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_CHAT_API_KEY")
    if not llm_key:
        print("[FATAL] LLM_API_KEY or OPENAI_CHAT_API_KEY environment variable is required.", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("LLM_API_KEY"):
        os.environ["LLM_API_KEY"] = llm_key

    hg_client = HugeGraphMemoryClient(graph=args.graph_name)
    backend = MemoryPipelineBackend(hg_client=hg_client)
    app = create_app(backend)

    # Serve static frontend files from the demo directory
    @app.route("/")
    def index():
        return send_from_directory(SCRIPT_DIR, "memory_frontend.html")

    @app.route("/hugegraph-memory-demo")
    def hugegraph_memory_demo():
        return send_from_directory(SCRIPT_DIR, "hugegraph-memory-demo.html")

    @app.route("/dashboard")
    def dashboard():
        return send_from_directory(SCRIPT_DIR, "dashboard.html")

    actual_graph = backend.hg.client.cfg.graph_name if backend.hg and backend.hg.client and backend.hg.client.cfg else args.graph_name
    print(f"[INFO] HugeGraph Memory Demo Server running at http://{args.host}:{args.port}")
    print(f"[INFO] Dashboard: http://{args.host}:{args.port}/dashboard")
    print(f"[INFO] Connected to HugeGraph graph={actual_graph}")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
