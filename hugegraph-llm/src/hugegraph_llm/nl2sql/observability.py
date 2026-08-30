# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lightweight observability for the NL2SQL service.

Counters and latency stats are kept in-process and rendered in Prometheus text
format on ``GET /metrics`` (no extra dependency). Request audit lines go
through ``hugegraph_llm.utils.log`` so they land in the same log stream as the
rest of the service -- that is the compliance trail ("who called what, when,
and how long it took").
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Dict

from hugegraph_llm.utils.log import log


class Metrics:
    """Thread-safe counters + latency aggregations keyed by route."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._latency_sum: Dict[str, float] = defaultdict(float)
        self._latency_count: Dict[str, int] = defaultdict(int)

    def inc(self, route: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[route] += amount

    def observe(self, route: str, seconds: float) -> None:
        with self._lock:
            self._latency_sum[route] += seconds
            self._latency_count[route] += 1

    def render(self) -> str:
        """Prometheus text exposition format (0.0.4)."""
        with self._lock:
            lines = ["# HELP nl2sql_requests_total Requests by route",
                     "# TYPE nl2sql_requests_total counter"]
            for route in sorted(self._counters):
                lines.append(
                    f'nl2sql_requests_total{{route="{route}"}} '
                    f"{self._counters[route]}"
                )
            lines += ["# HELP nl2sql_latency_seconds Latency by route",
                      "# TYPE nl2sql_latency_seconds summary"]
            for route in sorted(self._latency_count):
                n = self._latency_count[route]
                s = self._latency_sum[route]
                lines.append(
                    f'nl2sql_latency_seconds_sum{{route="{route}"}} {s:.6f}'
                )
                lines.append(
                    f'nl2sql_latency_seconds_count{{route="{route}"}} {n}'
                )
                lines.append(
                    f'nl2sql_latency_seconds_avg{{route="{route}"}} '
                    f"{s / max(n, 1):.6f}"
                )
            return "\n".join(lines) + "\n"


METRICS = Metrics()


def audit(method: str, path: str, status: int, seconds: float, client: str) -> None:
    """Write one audit line for an NL2SQL request."""
    log.info(
        "nl2sql audit: %s %s -> %s in %.1fms from %s",
        method, path, status, seconds * 1000, client,
    )
