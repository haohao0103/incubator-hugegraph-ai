#!/bin/bash
# Launcher for LightRAG Benchmark - loads API key from env or .env
set -e

cd "$(dirname "$0")"

# Load .env if exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Set API key if not already set (from poc_graphrag_bench_p0_v5.py)
if [ -z "$XIAOMI_MIMO_API_KEY" ] || [ "$XIAOMI_MIMO_API_KEY" = "" ]; then
    export XIAOMI_MIMO_API_KEY="REDACTED_API_KEY"
fi
export XIAOMI_MIMO_URL="https://api.xiaomimimo.com/v1"

echo "[Launcher] API Key: ${XIAOMI_MIMO_API_KEY:0:8}..."
echo "[Launcher] Starting: $*"

exec /Users/mac/.workbuddy/binaries/python/envs/hg-llm/bin/python3.10 "$@"
