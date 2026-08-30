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
Pluggable graph compute engines for the NL2SQL stack.

``LocalEngine`` is the default and the reference implementation; every NL2SQL
layer falls back to it when no engine is injected, so nothing here is required
to use the stack.

``VermeerEngine`` and ``VermeerClient`` are imported lazily: they talk HTTP to
a Vermeer cluster and pull in ``requests``, and there is no reason to pay that
import — or to fail on a missing dependency — for callers that only ever run
in-process.
"""

from typing import TYPE_CHECKING, Any, List

from .base import EngineCapabilities, GraphEngine, affinity_of
from .local import LocalEngine

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .vermeer import VermeerEngine
    from .vermeer_client import VermeerClient, VermeerError

_LAZY = {
    "VermeerEngine": ".vermeer",
    "VermeerClient": ".vermeer_client",
    "VermeerError": ".vermeer_client",
}

__all__ = [
    "EngineCapabilities",
    "GraphEngine",
    "LocalEngine",
    "VermeerClient",
    "VermeerEngine",
    "VermeerError",
    "affinity_of",
]


def __getattr__(name: str) -> Any:
    """Resolve Vermeer symbols on first access (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__() -> List[str]:
    return sorted(__all__)
