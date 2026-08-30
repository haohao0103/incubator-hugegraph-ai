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

"""
Minimal REST client for a Vermeer master.

Scope is deliberately narrow: load a graph, run a compute task, poll it, drop
the graph. Nothing else is needed by the NL2SQL engine.

Two non-obvious behaviours are baked in because both cost real debugging time:

1. **Proxy bypass.** ``requests`` honours ``HTTP_PROXY`` by default, and a
   localhost Vermeer behind a corporate/dev proxy answers ``502`` for every
   call — including ``/healthcheck``, which then looks like a *successful*
   "service is up" probe if you only check that the response is non-empty.
   The session sets ``trust_env = False`` so environment proxies never apply.

2. **Worker-group allocation.** Vermeer maps an unallocated task to the
   pseudo-group ``"$"``, while a worker registers under the group named in its
   ini (``worker_group=default``). The waiting-task scheduler only dispatches
   when the task's group equals an *idle worker group*, so ``"$" != "default"``
   means the task sits in ``waiting`` forever with no error anywhere.
   :meth:`VermeerClient.alloc_worker_group` binds space -> group and must be
   called once before submitting tasks.

API shape verified against Vermeer 1.x (``apps/master/services/router.go``).
"""

import json
import time
from typing import Any, Dict, Iterable, Optional

import requests

from hugegraph_llm.utils.log import log

#: Task states that mean "finished successfully".
LOAD_DONE_STATES = ("loaded", "on_disk")
COMPUTE_DONE_STATES = ("complete", "completed")
#: Task states that mean "will never finish".
FAILED_STATES = ("error", "canceled", "cancelled")


class VermeerError(RuntimeError):
    """Raised when Vermeer rejects a request or a task ends in error."""


class VermeerClient:
    """Thin, synchronous client for the Vermeer master HTTP API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:6688",
        space: str = "$DEFAULT",
        worker_group: str = "default",
        timeout: float = 30.0,
        poll_interval: float = 0.5,
        task_timeout: float = 600.0,
        session: Optional[requests.Session] = None,
    ):
        """
        :param base_url: Vermeer master HTTP endpoint (``http_peer`` in the
                         master ini; default port 6688).
        :param space: Vermeer space name. ``$DEFAULT`` is the built-in space
                      (``structure.DefaultSpaceName``) — the literal dollar
                      sign is correct, not a shell-escaping accident.
        :param worker_group: worker group to bind to ``space``.
        :param timeout: per-request timeout in seconds.
        :param poll_interval: task polling interval in seconds.
        :param task_timeout: give up on a task after this many seconds.
        """
        self._base = base_url.rstrip("/")
        self._space = space
        self._worker_group = worker_group
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._task_timeout = task_timeout
        if session is None:
            session = requests.Session()
            # See module docstring, point 1.
            session.trust_env = False
        self._session = session

    # ---- properties ----

    @property
    def base_url(self) -> str:
        return self._base

    @property
    def space(self) -> str:
        return self._space

    # ---- low level ----

    def _url(self, path: str) -> str:
        return f"{self._base}/{path.lstrip('/')}"

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = self._url(path)
        try:
            resp = self._session.request(
                method, url, json=payload, timeout=self._timeout,
                proxies={"http": None, "https": None},
            )
        except requests.RequestException as exc:
            raise VermeerError(f"{method} {url} failed: {exc}") from exc
        if resp.status_code >= 400:
            raise VermeerError(
                f"{method} {url} -> HTTP {resp.status_code}: {resp.text[:400]}"
            )
        text = resp.text.strip()
        if not text:
            return {}
        try:
            body = resp.json()
        except json.JSONDecodeError as exc:
            raise VermeerError(
                f"{method} {url} returned non-JSON: {text[:200]}"
            ) from exc
        if isinstance(body, dict) and body.get("errcode") not in (None, 0):
            raise VermeerError(
                f"{method} {url} -> errcode={body.get('errcode')} "
                f"message={body.get('message')}"
            )
        return body if isinstance(body, dict) else {"data": body}

    # ---- cluster ----

    def healthcheck(self) -> bool:
        """True when the master answers ``{"code": 200}``."""
        try:
            body = self._request("GET", "/healthcheck")
        except VermeerError as exc:
            log.debug("vermeer healthcheck failed: %s", exc)
            return False
        return int(body.get("code", 0)) == 200

    def workers(self) -> list:
        body = self._request("GET", "/workers")
        workers = body.get("workers", body.get("data", []))
        return workers if isinstance(workers, list) else []

    def worker_hosts(self, group: Optional[str] = None) -> list:
        """Distinct worker IPs, optionally restricted to a worker group.

        Local-file loading dispatches each file to the worker at a given IP
        (``load.vertex_files`` is an ``{ip: path}`` map), so the caller needs
        to know which hosts will read the files it just wrote.
        """
        want = self._worker_group if group is None else group
        hosts = []
        for w in self.workers():
            if not isinstance(w, dict):
                continue
            ip = w.get("ip_addr") or w.get("IpAddr")
            if not ip:
                continue
            if want and w.get("group", want) != want:
                continue
            if ip not in hosts:
                hosts.append(ip)
        return hosts

    def alloc_worker_group(self) -> None:
        """Bind ``worker_group`` to ``space``. Required before any task."""
        self._request(
            "POST",
            f"/admin/workers/alloc/{self._worker_group}/{self._space}",
        )
        log.debug("vermeer worker group %r allocated to space %r",
                  self._worker_group, self._space)

    # ---- graphs ----

    def create_graph(self, name: str) -> None:
        """Declare a graph.

        Vermeer's handler is effectively a no-op — the graph materialises when
        the load task runs — but calling it keeps the sequence explicit and
        surfaces auth/route problems before a long load.
        """
        self._request("POST", "/graphs/create", {"name": name})

    def delete_graph(self, name: str) -> None:
        try:
            self._request("DELETE", f"/graphs/{name}")
        except VermeerError as exc:
            log.debug("vermeer delete graph %s failed (ignored): %s", name, exc)

    def graph_names(self) -> list:
        body = self._request("GET", "/graphs")
        graphs = body.get("graphs", body.get("data", []))
        names = []
        for g in graphs if isinstance(graphs, list) else []:
            if isinstance(g, dict) and g.get("name"):
                names.append(g["name"])
        return names

    # ---- tasks ----

    def submit_task(
        self, task_type: str, graph: str, params: Dict[str, str]
    ) -> int:
        """Create a task and return its id.

        :param task_type: ``"load"`` or ``"compute"``.
        :param graph: graph name.
        :param params: Vermeer task params; every value must be a string.
        """
        body = self._request(
            "POST",
            "/tasks/create",
            {
                "task_type": task_type,
                "graph": graph,
                "params": {k: str(v) for k, v in params.items()},
            },
        )
        task = body.get("task") or {}
        task_id = task.get("id")
        if task_id is None:
            raise VermeerError(f"no task id in response: {body}")
        log.debug("vermeer %s task %s submitted on graph %s",
                  task_type, task_id, graph)
        return int(task_id)

    def task(self, task_id: int) -> Dict[str, Any]:
        body = self._request("GET", f"/task/{task_id}")
        return body.get("task") or body

    def task_state(self, task_id: int) -> str:
        return str(self.task(task_id).get("state", ""))

    def wait_task(
        self,
        task_id: int,
        done_states: Iterable[str],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Poll until the task reaches one of ``done_states``.

        :raises VermeerError: on an error state or on timeout.
        """
        done = tuple(done_states)
        deadline = time.time() + (
            self._task_timeout if timeout is None else timeout
        )
        last_state = ""
        while True:
            info = self.task(task_id)
            state = str(info.get("state", ""))
            if state != last_state:
                log.debug("vermeer task %s state: %s", task_id, state)
                last_state = state
            if state in done:
                return info
            if state in FAILED_STATES:
                raise VermeerError(
                    f"task {task_id} ended in {state}: "
                    f"{info.get('error_msg') or info.get('message') or ''}"
                )
            if time.time() >= deadline:
                raise VermeerError(
                    f"task {task_id} still {state!r} after timeout"
                )
            time.sleep(self._poll_interval)

    # ---- composites ----

    def compute_values(self, task_id: int, limit: int = 100000) -> Dict[str, str]:
        """Fetch a finished compute task's vertex values as ``{id: value}``.

        Requires the task to have been submitted with ``output.need_query=1``;
        otherwise the master drops the result set one minute after completion
        and this returns an empty dict.

        Pagination follows the master's cursor protocol: ``/tasks/value/{id}``
        answers ``{"vertices": [...], "cursor": n}`` and signals exhaustion by
        ``message == "EOF"`` (an empty page, not an error).
        """
        limit = max(1, min(int(limit), 100000))
        values: Dict[str, str] = {}
        cursor = 0
        while True:
            body = self._request(
                "GET", f"/tasks/value/{task_id}?cursor={cursor}&limit={limit}"
            )
            rows = body.get("vertices") or []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                # The Go struct carries no json tags, so the wire format is
                # {"ID": ..., "Value": ...}; tolerate the lowercase variant.
                node_id = row.get("ID", row.get("id"))
                value = row.get("Value", row.get("value"))
                if node_id is not None:
                    values[str(node_id)] = "" if value is None else str(value)
            if str(body.get("message", "")).upper() == "EOF" or not rows:
                break
            nxt = body.get("cursor")
            if nxt is None or int(nxt) <= cursor:
                break
            cursor = int(nxt)
        log.debug("vermeer task %s returned %s values", task_id, len(values))
        return values

    def run_load(self, graph: str, params: Dict[str, str]) -> Dict[str, Any]:
        task_id = self.submit_task("load", graph, params)
        return self.wait_task(task_id, LOAD_DONE_STATES)

    def run_compute(self, graph: str, params: Dict[str, str]) -> Dict[str, Any]:
        task_id = self.submit_task("compute", graph, params)
        return self.wait_task(task_id, COMPUTE_DONE_STATES)

    def close(self) -> None:
        self._session.close()
