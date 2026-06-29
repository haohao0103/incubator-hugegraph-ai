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
Download third-party benchmark datasets required by HugeGraph-AI-Memory tests.

These datasets are intentionally excluded from git (see root .gitignore) to
avoid redistributing data with unclear/strict licenses. Run this script before
executing the corresponding benchmark runners.

Usage:
    python tests/download_datasets.py
    python tests/download_datasets.py --dataset locomo
    python tests/download_datasets.py --dataset icews
"""

import argparse
import os
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

# LOCOMO: Snap Research, permissive for research use.
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/LoCoMo/main/data/locomo10.json"
LOCOMO_DIR = SCRIPT_DIR / "locomo_data"
LOCOMO_FILE = LOCOMO_DIR / "locomo10.json"

# ICEWS14 agent-memory benchmark file.
# This is a derived benchmark file built from ICEWS14 (CMU). Because the
# original ICEWS14 terms require accepting a license on the Dataverse page,
# we do not auto-download it here. Two options:
#   1. Obtain icews14_agent_memory_benchmark.json from the project maintainer
#      and place it at tests/benchmark_data/icews14_agent_memory_benchmark.json
#   2. Build it from the original ICEWS14 raw files (train/test/valid.txt)
#      using your own pipeline.
# You may override the source URL with the ICEWS_BENCHMARK_URL env var.
ICEWS_BENCHMARK_URL = os.environ.get(
    "ICEWS_BENCHMARK_URL",
    "",  # No default public mirror; populate manually or via project mirror.
)
ICEWS_DIR = SCRIPT_DIR / "benchmark_data"
ICEWS_FILE = ICEWS_DIR / "icews14_agent_memory_benchmark.json"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved {dest.stat().st_size} bytes to {dest}")


def download_locomo() -> Path:
    if LOCOMO_FILE.exists():
        print(f"LOCOMO already exists: {LOCOMO_FILE}")
        return LOCOMO_FILE
    _download(LOCOMO_URL, LOCOMO_FILE)
    return LOCOMO_FILE


def download_icews() -> Path:
    if ICEWS_FILE.exists():
        print(f"ICEWS benchmark already exists: {ICEWS_FILE}")
        return ICEWS_FILE
    if not ICEWS_BENCHMARK_URL:
        raise RuntimeError(
            "No ICEWS_BENCHMARK_URL configured.\n"
            "Please obtain 'icews14_agent_memory_benchmark.json' from the project maintainer\n"
            "or set ICEWS_BENCHMARK_URL to a trusted mirror.\n"
            "Place the file at: " + str(ICEWS_FILE)
        )
    _download(ICEWS_BENCHMARK_URL, ICEWS_FILE)
    return ICEWS_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--dataset",
        choices=["locomo", "icews", "all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    args = parser.parse_args()

    if args.dataset in ("locomo", "all"):
        download_locomo()
    if args.dataset in ("icews", "all"):
        download_icews()


if __name__ == "__main__":
    main()
