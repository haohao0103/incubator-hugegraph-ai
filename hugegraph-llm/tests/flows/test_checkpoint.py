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

"""Tests for IncrementalCheckpointManager and CheckpointingNode."""

import json
import os
import tempfile

import pytest

from hugegraph_llm.flows.checkpoint import (
    INCREMENTAL_STAGES,
    IncrementalCheckpointManager,
    StageCheckpoint,
)


def test_generate_job_id():
    manager = IncrementalCheckpointManager(checkpoint_dir="/tmp")
    assert manager.job_id.startswith("job_")
    assert len(manager.job_id) > 20


def test_manifest_created():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        assert os.path.isdir(manager.job_dir)
        assert not os.path.exists(manager.checkpoint_path)
        # Loading a non-existent manifest returns defaults.
        manifest = manager._load_manifest()
        assert manifest["job_id"] == "j1"
        for stage in INCREMENTAL_STAGES:
            assert stage in manifest["stages"]


def test_save_and_load_stage():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        manager.save_stage("chunk_split", "completed", {"chunks": ["a", "b"]})

        assert manager.get_stage_status("chunk_split") == "completed"
        state = manager.load_stage_state("chunk_split")
        assert state == {"chunks": ["a", "b"]}

        manifest = manager._load_manifest()
        assert manifest["stages"]["chunk_split"]["state_path"] is not None


def test_atomic_write_uses_temp():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        manager.save_stage("chunk_split", "completed", {"chunks": ["a"]})
        # No leftover temp files.
        temp_files = [f for f in os.listdir(manager.job_dir) if f.endswith(".tmp")]
        assert temp_files == []


def test_get_resume_stage_from_scratch():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        assert manager.get_resume_stage() == "chunk_split"


def test_get_resume_stage_after_completion():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        manager.save_stage("chunk_split", "completed", {"chunks": ["a"]})
        assert manager.get_resume_stage() == "extract_info"


def test_get_resume_stage_all_completed():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        for stage in INCREMENTAL_STAGES:
            manager.save_stage(stage, "completed", {"done": stage})
        assert manager.get_resume_stage() is None


def test_get_resume_stage_retry_failed():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        manager.save_stage("chunk_split", "completed", {"chunks": ["a"]})
        manager.save_stage("extract_info", "failed", error="llm error")
        assert manager.get_resume_stage() == "extract_info"
        assert manager.get_stage_error("extract_info") == "llm error"


def test_get_resume_stage_retry_running():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        manager.save_stage("chunk_split", "completed", {"chunks": ["a"]})
        manager.save_stage("extract_info", "running", {"chunks": ["a"]})
        assert manager.get_resume_stage() == "extract_info"


def test_summary():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        manager.save_stage("chunk_split", "completed", {"chunks": ["a"]})
        manager.save_stage("extract_info", "failed", error="boom")
        summary = manager.get_summary()
        assert summary["job_id"] == "j1"
        assert summary["completed_stages"] == ["chunk_split"]
        assert summary["failed_stages"] == ["extract_info"]
        assert summary["next_stage"] == "extract_info"


def test_stage_checkpoint_from_dict():
    sc = StageCheckpoint.from_dict({"stage": "s", "status": "completed"})
    assert sc.stage == "s"
    assert sc.status == "completed"


def test_save_stage_failed_clears_state_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        manager.save_stage("chunk_split", "running", {"chunks": ["a"]})
        manager.save_stage("chunk_split", "failed", error="boom")
        assert manager.get_stage_status("chunk_split") == "failed"
        assert manager.get_stage_error("chunk_split") == "boom"
        manifest = manager._load_manifest()
        assert manifest["stages"]["chunk_split"].get("state_path") is None


def test_manifest_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        manager.save_stage("chunk_split", "completed", {"chunks": ["a"]})
        # Re-create manager from the same directory.
        manager2 = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        assert manager2.get_stage_status("chunk_split") == "completed"
        assert manager2.load_stage_state("chunk_split") == {"chunks": ["a"]}


def test_load_stage_state_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        assert manager.load_stage_state("nonexistent") is None


def test_load_stage_state_missing_file_returns_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        # Manifest claims a state path but the file has been deleted.
        manager._save_manifest({
            "job_id": "j1",
            "created_at": 0,
            "stages": {
                "chunk_split": {
                    "stage": "chunk_split",
                    "status": "completed",
                    "timestamp": 0,
                    "state_path": os.path.join(tmpdir, "missing.json"),
                }
            },
        })
        assert manager.load_stage_state("chunk_split") is None
