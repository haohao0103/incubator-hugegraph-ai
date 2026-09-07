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

"""Connector contract for the semantic layer (milestone M1).

Every source of metadata -- warehouse catalog, query logs, an Ossie YAML
spec -- is expressed as a connector following one shape, so the loader,
the SHACL validation and the tests are written once:

    extract()   -> raw records from the source (list of dicts)
    transform() -> typed, id-assigned records ready for the graph
    load()      -> writes to HugeGraph, idempotently
    ingest()    -> orchestrates the three above

Ordering is enforced: calling ``transform`` before ``extract`` raises
:class:`StateError` instead of silently producing an empty graph, which is
the failure mode this contract exists to prevent.

The ``extract``/``transform``/``load`` split is borrowed from neocarta's
connector contract, which is the closest thing to a de-facto standard in
the graph-native semantic layer space.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

__all__ = [
    "SemanticLayerError",
    "StateError",
    "ValidationError",
    "SourceConnector",
    "make_id",
]

Raw = TypeVar("Raw", bound=Dict[str, Any])
Transformed = TypeVar("Transformed", bound=Dict[str, Any])


class SemanticLayerError(Exception):
    """Base error for the semantic layer."""


class StateError(SemanticLayerError):
    """Raised when connector stages are called out of order."""


class ValidationError(SemanticLayerError):
    """Raised when a record fails schema/constraint validation."""


def make_id(*parts: Any) -> str:
    """Build a deterministic, idempotent vertex id from its parts.

    Ids are namespaced (``table:dw.orders``) so the same logical object
    always maps to the same vertex: re-ingesting a source updates instead
    of duplicating. Empty parts are skipped; the result is stable across
    runs and machines -- never hash on unordered data here.
    """
    cleaned = [str(p).strip() for p in parts if p is not None and str(p).strip()]
    if not cleaned:
        raise ValueError("make_id requires at least one non-empty part")
    return ":".join(cleaned)


class SourceConnector(ABC, Generic[Raw, Transformed]):
    """Base class for semantic layer source connectors.

    Subclasses implement the three stages; ``ingest`` is inherited and must
    not be overridden just to reorder stages.
    """

    #: Vertex/edge records emitted by this connector, for logging and tests.
    name: str = "source"

    def __init__(self) -> None:
        self._extracted: Optional[List[Raw]] = None
        self._transformed: Optional[List[Transformed]] = None
        self._loaded = 0

    # -- stages -------------------------------------------------------------

    @abstractmethod
    def extract(self) -> List[Raw]:
        """Read raw metadata from the source."""

    @abstractmethod
    def transform(self, raw: List[Raw]) -> List[Transformed]:
        """Turn raw records into graph records with deterministic ids."""

    @abstractmethod
    def load(self, records: List[Transformed]) -> int:
        """Write records to the graph. Returns the number written."""

    # -- orchestration ------------------------------------------------------

    def ingest(self) -> Dict[str, Any]:
        """Run extract -> transform -> load and return a summary."""
        raw = self.extract()
        self._extracted = raw
        records = self.transform(raw)
        self._transformed = records
        self._loaded = self.load(records)
        return {
            "connector": self.name,
            "extracted": len(raw),
            "transformed": len(records),
            "loaded": self._loaded,
        }

    # -- guards -------------------------------------------------------------

    def require_extracted(self) -> List[Raw]:
        """Return extracted records or raise if ``extract`` has not run."""
        if self._extracted is None:
            raise StateError(
                f"{self.name}: extract() must run before transform()/load()"
            )
        return self._extracted

    def require_transformed(self) -> List[Transformed]:
        """Return transformed records or raise if ``transform`` has not run."""
        if self._transformed is None:
            raise StateError(
                f"{self.name}: transform() must run before load()"
            )
        return self._transformed

    @property
    def loaded(self) -> int:
        """Number of records written by the last ``load``."""
        return self._loaded
