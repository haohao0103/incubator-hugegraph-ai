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

"""Checkpoint / resume support for incremental indexing flows.

Design borrowed from LightRAG's DocStatusStorage (per-document status) and
MS-GraphRAG's PipelineRunContext (state snapshot after each workflow).  The
IncrementalCheckpointManager persists the workflow state after each completed
stage so that a crashed or interrupted incremental index run can resume from the
last successful stage instead of re-running from the beginning.

Checkpoint file layout (under checkpoint_dir):
    {job_id}/checkpoint.json       -- stage status and metadata
    {job_id}/{stage}_state.json    -- WkFlowState snapshot for that stage

Atomicity: each write is performed to a temp file and renamed into place.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hugegraph_llm.utils.log import log


# Stage order for the incremental indexing pipeline.  The manager uses this list
# to determine which stage should run next after a restart.
INCREMENTAL_STAGES: List[str] = [
    "auto_schema_kg",
    "chunk_split",
    "extract_info",
    "entity_resolution",
    "commit_new",
    "affected_detect",
    "incremental_report",
    "incremental_vector",
]


@dataclass
class StageCheckpoint:
    """Status of a single stage within a checkpointed job."""

    stage: str
    status: str = "pending"  # pending | running | completed | failed
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None
    state_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "timestamp": self.timestamp,
            "error": self.error,
            "state_path": self.state_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageCheckpoint":
        return cls(
            stage=data.get("stage", ""),
            status=data.get("status", "pending"),
            timestamp=data.get("timestamp", time.time()),
            error=data.get("error"),
            state_path=data.get("state_path"),
        )


class IncrementalCheckpointManager:
    """Manages persistent checkpoints for an incremental indexing job.

    Example:
        manager = IncrementalCheckpointManager(
            checkpoint_dir="/tmp/hg_checkpoints",
            job_id="job_20260702_001",
        )
        manager.save_stage("chunk_split", "completed", state.to_json())
        next_stage = manager.get_resume_stage()
    """

    def __init__(
        self,
        checkpoint_dir: str,
        job_id: Optional[str] = None,
        stages: Optional[List[str]] = None,
    ):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.job_id = job_id or self._generate_job_id()
        self.stages = list(stages or INCREMENTAL_STAGES)
        self.job_dir = os.path.join(self.checkpoint_dir, self.job_id)
        self.checkpoint_path = os.path.join(self.job_dir, "checkpoint.json")
        os.makedirs(self.job_dir, exist_ok=True)

    @staticmethod
    def _generate_job_id() -> str:
        return f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def _state_path(self, stage: str) -> str:
        return os.path.join(self.job_dir, f"{stage}_state.json")

    def _load_manifest(self) -> Dict[str, Any]:
        if not os.path.exists(self.checkpoint_path):
            return {
                "job_id": self.job_id,
                "created_at": time.time(),
                "stages": {stage: StageCheckpoint(stage).to_dict() for stage in self.stages},
            }
        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        tmp_path = self.checkpoint_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.checkpoint_path)

    def save_stage(
        self,
        stage: str,
        status: str,
        state_dict: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Persist the status of a stage and optionally its state snapshot.

        Args:
            stage: Stage name.
            status: One of pending, running, completed, failed.
            state_dict: WkFlowState snapshot (only stored for completed stages).
            error: Error message for failed stages.
        """
        manifest = self._load_manifest()
        stages = manifest.setdefault("stages", {})
        stage_record = stages.get(stage, StageCheckpoint(stage).to_dict())
        stage_record["status"] = status
        stage_record["timestamp"] = time.time()
        stage_record["error"] = error
        if status in ("running", "completed") and state_dict is not None:
            state_path = self._state_path(stage)
            tmp_path = state_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, state_path)
            stage_record["state_path"] = state_path
        elif status == "failed":
            stage_record["state_path"] = None
        stages[stage] = stage_record
        self._save_manifest(manifest)
        log.info(
            "Checkpoint %s: stage=%s status=%s", self.job_id, stage, status
        )

    def get_stage_status(self, stage: str) -> str:
        manifest = self._load_manifest()
        return manifest.get("stages", {}).get(stage, {}).get("status", "pending")

    def get_stage_error(self, stage: str) -> Optional[str]:
        manifest = self._load_manifest()
        return manifest.get("stages", {}).get(stage, {}).get("error")

    def get_resume_stage(self) -> Optional[str]:
        """Return the stage that should run next.

        If all stages are completed, returns None.  If a stage previously failed
        or was running, that stage is returned so it can be retried.
        """
        manifest = self._load_manifest()
        stages_status = {s: manifest.get("stages", {}).get(s, {}).get("status", "pending")
                         for s in self.stages}
        # Resume a failed or interrupted (running) stage first.
        for stage in self.stages:
            if stages_status.get(stage) in ("failed", "running"):
                return stage
        # Otherwise continue after the last completed stage.
        last_completed_idx = -1
        for idx, stage in enumerate(self.stages):
            if stages_status.get(stage) == "completed":
                last_completed_idx = idx
        next_idx = last_completed_idx + 1
        if next_idx < len(self.stages):
            return self.stages[next_idx]
        return None

    def load_stage_state(self, stage: str) -> Optional[Dict[str, Any]]:
        """Load the state snapshot associated with a completed stage."""
        manifest = self._load_manifest()
        stage_record = manifest.get("stages", {}).get(stage, {})
        state_path = stage_record.get("state_path")
        if state_path and os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Return a human-readable summary of the checkpointed job."""
        manifest = self._load_manifest()
        stages = manifest.get("stages", {})
        completed = [s for s in self.stages if stages.get(s, {}).get("status") == "completed"]
        failed = [s for s in self.stages if stages.get(s, {}).get("status") == "failed"]
        running = [s for s in self.stages if stages.get(s, {}).get("status") == "running"]
        return {
            "job_id": self.job_id,
            "checkpoint_dir": self.checkpoint_dir,
            "created_at": manifest.get("created_at"),
            "completed_stages": completed,
            "failed_stages": failed,
            "running_stages": running,
            "next_stage": self.get_resume_stage(),
        }
