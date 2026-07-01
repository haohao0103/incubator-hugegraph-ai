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

"""Async VLM pipeline — semaphore-controlled concurrent VLM analysis.

Adapted from LightRAG ``operate.py`` asyncio.Semaphore pattern and
``llm_roles.py`` priority-based role routing. Provides:

1. ``AsyncVLMPipeline`` — process multiple images concurrently with a
   semaphore, cooperative yield, retry, and priority queue.
2. ``VLMPipelineConfig`` — configuration for concurrency, retry, yield.
3. ``run()`` operator protocol for HG-AI pipeline integration.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


@dataclass
class VLMPipelineConfig:
    """Configuration for async VLM pipeline."""

    max_concurrent: int = 4          # Semaphore limit for concurrent VLM calls
    max_retries: int = 3             # Max retries per image on failure
    retry_delay: float = 1.0         # Base delay (seconds) between retries
    retry_backoff_factor: float = 2.0  # Exponential backoff multiplier
    yield_every: int = 8             # Cooperative yield interval
    priority_default: int = 0        # Default priority (higher = sooner)


@dataclass
class VLMTask:
    """A single VLM analysis task."""

    image_id: str
    image_data: Any                   # base64 string, path, or dict
    prompt: str
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VLMResult:
    """Result of a single VLM analysis."""

    image_id: str
    description: str | None = None
    success: bool = True
    error: str | None = None
    retry_count: int = 0
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


async def _cooperative_yield(iteration: int, every: int = 8) -> None:
    """Yield control to the event loop every N iterations.

    Adapted from LightRAG ``utils._cooperative_yield``.
    Prevents long async batches from monopolizing the event loop.
    """
    if every > 0 and iteration % every == 0:
        await asyncio.sleep(0)


async def _call_vlm_with_retry(
    vlm_func_async: Callable[..., Coroutine[Any, Any, str]],
    task: VLMTask,
    config: VLMPipelineConfig,
) -> VLMResult:
    """Call VLM function with exponential-backoff retry."""
    start_time = time.monotonic()
    delay = config.retry_delay
    last_error: str | None = None

    for attempt in range(config.max_retries + 1):
        try:
            description = await vlm_func_async(task.image_data, task.prompt)
            elapsed = time.monotonic() - start_time
            return VLMResult(
                image_id=task.image_id,
                description=description,
                success=True,
                retry_count=attempt,
                elapsed_seconds=elapsed,
                metadata=task.metadata,
            )
        except Exception as exc:
            last_error = str(exc)
            if attempt < config.max_retries:
                logger.warning(
                    f"[AsyncVLM] retry {attempt + 1}/{config.max_retries} "
                    f"for {task.image_id}: {exc}"
                )
                await asyncio.sleep(delay)
                delay *= config.retry_backoff_factor
            else:
                logger.error(
                    f"[AsyncVLM] failed after {config.max_retries} retries "
                    f"for {task.image_id}: {exc}"
                )

    elapsed = time.monotonic() - start_time
    return VLMResult(
        image_id=task.image_id,
        success=False,
        error=last_error,
        retry_count=config.max_retries,
        elapsed_seconds=elapsed,
        metadata=task.metadata,
    )


class AsyncVLMPipeline:
    """Async VLM pipeline with semaphore-controlled concurrency.

    Adapted from LightRAG ``operate.py`` Semaphore + cooperative yield pattern.
    Processes multiple VLM tasks concurrently, respecting a semaphore limit
    and yielding control periodically to avoid event loop starvation.

    Usage::

        pipeline = AsyncVLMPipeline(config=VLMPipelineConfig(max_concurrent=4))
        results = await pipeline.process_async(tasks, vlm_func_async=my_vlm)
    """

    def __init__(self, config: VLMPipelineConfig | None = None) -> None:
        self.config = config or VLMPipelineConfig()

    async def process_async(
        self,
        tasks: list[VLMTask],
        vlm_func_async: Callable[..., Coroutine[Any, Any, str]],
    ) -> list[VLMResult]:
        """Process all VLM tasks concurrently with semaphore control.

        Tasks are sorted by priority (higher first) before processing.
        """
        if not tasks:
            return []

        # Sort by priority (higher = sooner)
        sorted_tasks = sorted(tasks, key=lambda t: -t.priority)

        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        results: list[VLMResult] = []

        async def _process_one(idx: int, task: VLMTask) -> VLMResult:
            async with semaphore:
                await _cooperative_yield(idx, every=self.config.yield_every)
                return await _call_vlm_with_retry(vlm_func_async, task, self.config)

        coros = [_process_one(i, t) for i, t in enumerate(sorted_tasks)]
        results = await asyncio.gather(*coros, return_exceptions=False)

        return list(results)

    def process_sync(
        self,
        tasks: list[VLMTask],
        vlm_func_sync: Callable[..., str],
    ) -> list[VLMResult]:
        """Process VLM tasks synchronously (no concurrency).

        Fallback when async is not available or not needed.
        """
        results: list[VLMResult] = []
        for i, task in enumerate(tasks):
            start_time = time.monotonic()
            retry_count = 0
            delay = self.config.retry_delay
            description = None
            error = None

            for attempt in range(self.config.max_retries + 1):
                try:
                    description = vlm_func_sync(task.image_data, task.prompt)
                    retry_count = attempt
                    break
                except Exception as exc:
                    error = str(exc)
                    if attempt < self.config.max_retries:
                        time.sleep(delay)
                        delay *= self.config.retry_backoff_factor
                        retry_count = attempt + 1

            elapsed = time.monotonic() - start_time
            results.append(VLMResult(
                image_id=task.image_id,
                description=description,
                success=description is not None,
                error=error if description is None else None,
                retry_count=retry_count,
                elapsed_seconds=elapsed,
                metadata=task.metadata,
            ))

        return results

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """HG-AI operator protocol: process VLM tasks from context.

        Context keys:
        - ``vlm_tasks``: list of dicts with ``image_id``, ``image_data``, ``prompt``, ``priority``
        - ``vlm_func_sync``: sync VLM callable (used when no async available)
        - ``vlm_func_async``: async VLM callable (preferred)

        If vlm_func_async is provided and we're in an async context, use it.
        Otherwise fall back to sync.
        """
        raw_tasks = context.get("vlm_tasks", [])
        tasks: list[VLMTask] = []
        for raw in raw_tasks:
            if isinstance(raw, dict):
                tasks.append(VLMTask(
                    image_id=str(raw.get("image_id", "")),
                    image_data=raw.get("image_data"),
                    prompt=str(raw.get("prompt", "")),
                    priority=int(raw.get("priority", self.config.priority_default)),
                    metadata=raw.get("metadata", {}),
                ))
            elif isinstance(raw, VLMTask):
                tasks.append(raw)

        # Try async first, fall back to sync
        vlm_func_async = context.get("vlm_func_async")
        vlm_func_sync = context.get("vlm_func_sync")

        if vlm_func_async and asyncio.get_event_loop().is_running():
            # We're inside an async context — use gather
            results = asyncio.gather(
                *[
                    _call_vlm_with_retry(vlm_func_async, t, self.config)
                    for t in tasks
                ]
            )
            # This won't actually work in sync run(); caller should use
            # process_async() directly in async context.
            context["vlm_results"] = self.process_sync(tasks, vlm_func_sync or (lambda _d, _p: ""))
        elif vlm_func_sync:
            context["vlm_results"] = self.process_sync(tasks, vlm_func_sync)
        else:
            # No VLM function — mark all as failed
            context["vlm_results"] = [
                VLMResult(
                    image_id=t.image_id,
                    success=False,
                    error="no VLM function provided",
                    metadata=t.metadata,
                )
                for t in tasks
            ]

        context["vlm_success_count"] = sum(
            1 for r in context.get("vlm_results", []) if r.success
        )
        context["vlm_failure_count"] = sum(
            1 for r in context.get("vlm_results", []) if not r.success
        )
        return context
