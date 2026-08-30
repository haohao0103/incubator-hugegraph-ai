"""Start (or stop) the demo Gradio server detached from the agent session.

WorkBuddy background bash tasks get cleaned up when a turn ends (process-group
kill), so the server is launched with ``start_new_session=True`` and its pid is
written to a file -- it survives across turns and can be managed explicitly.

Usage::

    start:  python start_demo_server.py [--port 8001] [--host 127.0.0.1]
    stop:   python start_demo_server.py --stop
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_REPO, "hugegraph-llm", "src")
_LOG = os.path.join(os.path.dirname(__file__), "logs", "app_server.log")
_PID = os.path.join(os.path.dirname(__file__), "logs", "app_server.pid")
_VENV = "/Users/mac/.workbuddy/binaries/python/envs/hg-e2e/bin/python"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def stop() -> int:
    if not os.path.exists(_PID):
        print("no pid file; nothing to stop")
        return 0
    pid = int(open(_PID).read().strip())
    if _alive(pid):
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        print(f"sent SIGTERM to server (pid {pid})")
    else:
        print(f"pid {pid} not alive")
    try:
        os.remove(_PID)
    except OSError:
        pass
    return 0


def start(host: str, port: int) -> int:
    os.makedirs(os.path.dirname(_LOG), exist_ok=True)
    log = open(_LOG, "a")
    cmd = [
        _VENV, "-m", "uvicorn",
        "hugegraph_llm.demo.rag_demo.app:create_app",
        "--host", host, "--port", str(port),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=_SRC,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    with open(_PID, "w") as f:
        f.write(str(proc.pid))
    # health poll
    import urllib.request

    url = f"http://{host}:{port}/"
    for _ in range(40):
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    print(f"PASS: demo server up at {url} (pid {proc.pid})")
                    print(f"  pid file: {_PID}")
                    print(f"  log file: {_LOG}")
                    return 0
        except Exception:
            time.sleep(1)
    print(f"FAIL: server did not answer at {url}; see {_LOG}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--stop", action="store_true", help="stop the running server")
    args = parser.parse_args()
    if args.stop:
        return stop()
    return start(args.host, args.port)


if __name__ == "__main__":
    sys.exit(main())
