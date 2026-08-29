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

"""Incremental merge utilities — LightRAG-style entity/relation merging.

Borrowed from LightRAG's ``merge_nodes_then_upsert`` and
``merge_edges_then_upsert`` (operate.py lines 2914-3318), these utilities
handle the merge phase of incremental document insertion:

1. **SourceIdsManager**: Track which document chunks contributed to an
   entity/relation, with KEEP/FIFO limit strategies (LightRAG's
   ``source_ids`` upper-limit of 200).
2. **vote_entity_type**: Majority-vote entity type using ``Counter.most_common``.
3. **merge_entity_descriptions**: Dedup + DescriptionMerger four-level merge.
4. **IncrementalMergePipeline**: Orchestrates entity→relation→chunk merge.

These utilities complement the existing ``incremental_utils.py`` (which
handles community assignment persistence and affected community detection)
and the ``key_lock.py`` (which provides per-key concurrency control).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from hugegraph_llm.operators.llm_op.description_merger import (
    DescriptionMerger,
    DescriptionMergerConfig,
)
from hugegraph_llm.utils.log import log

# ---------------------------------------------------------------------------
# Source IDs management (LightRAG's source_ids limit strategy)
# ---------------------------------------------------------------------------


class SourceIdStrategy(Enum):
    """How to handle source_ids when the limit is exceeded.

    Mirrors LightRAG's two strategies:
    - **KEEP**: Keep the earliest source_ids (preserve provenance).
    - **FIFO**: Keep the latest source_ids (freshness over provenance).
    """

    KEEP = "keep"    # Preserve earliest (original) sources
    FIFO = "fifo"    # Keep most recent (fresh) sources


@dataclass
class SourceIdsConfig:
    """Configuration for source IDs management."""

    max_source_ids: int = 200       # LightRAG's default upper limit
    strategy: SourceIdStrategy = SourceIdStrategy.KEEP


class SourceIdsManager:
    """Manage the list of source document IDs associated with an entity/relation.

    LightRAG limits source_ids to avoid unbounded growth.  When the limit
    is exceeded, the chosen strategy determines which IDs to keep.

    Usage::

        mgr = SourceIdsManager(config=SourceIdsConfig(max_source_ids=200))
        ids = mgr.merge(existing_ids=["doc1", "doc2"], new_ids=["doc3", "doc4"])
    """

    def __init__(self, config: Optional[SourceIdsConfig] = None) -> None:
        self.config = config or SourceIdsConfig()

    def merge(
        self,
        existing_ids: Optional[List[str]] = None,
        new_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Merge new source IDs into existing list with dedup and limit.

        Args:
            existing_ids: Current source_ids list (may be None or empty).
            new_ids: Source IDs from the new document chunk.

        Returns:
            Merged and limited source_ids list.
        """
        existing = list(existing_ids or [])
        new = list(new_ids or [])

        # Dedup while preserving order: existing first, then new additions
        merged = list(dict.fromkeys(existing + new))

        # Apply limit
        if len(merged) > self.config.max_source_ids:
            if self.config.strategy == SourceIdStrategy.KEEP:
                # Keep earliest IDs (first N)
                merged = merged[:self.config.max_source_ids]
            elif self.config.strategy == SourceIdStrategy.FIFO:
                # Keep latest IDs (last N)
                merged = merged[-self.config.max_source_ids:]

        return merged

    def remove(self, ids: List[str], to_remove: List[str]) -> List[str]:
        """Remove specific IDs from the source list."""
        remove_set = set(to_remove)
        return [id_ for id_ in ids if id_ not in remove_set]


# ---------------------------------------------------------------------------
# Entity type voting (LightRAG's Counter.most_common(1))
# ---------------------------------------------------------------------------


def vote_entity_type(
    existing_type: Optional[str] = None,
    new_types: Optional[List[str]] = None,
    all_types: Optional[List[str]] = None,
) -> str:
    """Resolve entity type conflicts by majority vote.

    Mirrors LightRAG's ``Counter.most_common(1)`` pattern: when multiple
    chunks contribute different entity_type labels for the same entity,
    the most frequent type wins.

    Args:
        existing_type: The current entity type (if any).
        new_types: Types from the new document chunks.
        all_types: If provided, overrides existing+new (full recount).

    Returns:
        The majority-voted entity type string.  Empty string if no types.
    """
    if all_types is not None:
        types = [t for t in all_types if t]
    else:
        types = [t for t in ([existing_type] + list(new_types or [])) if t]

    if not types:
        return ""

    counter = Counter(types)
    winner = counter.most_common(1)[0][0]
    return winner


# ---------------------------------------------------------------------------
# Description merge (using DescriptionMerger)
# ---------------------------------------------------------------------------


def merge_entity_descriptions(
    existing_description: Optional[str] = None,
    new_descriptions: Optional[List[str]] = None,
    merger: Optional[DescriptionMerger] = None,
) -> str:
    """Merge entity/relation descriptions using four-level strategy.

    Wraps ``DescriptionMerger`` with a convenient interface that handles
    existing + new descriptions.

    Args:
        existing_description: The current entity description (if any).
        new_descriptions: New descriptions from chunks.
        merger: Pre-configured DescriptionMerger (created if None).

    Returns:
        Merged description string.
    """
    if merger is None:
        merger = DescriptionMerger()

    all_descs: List[str] = []
    if existing_description:
        all_descs.append(existing_description)
    all_descs.extend(new_descriptions or [])

    return merger.merge(all_descs)


# ---------------------------------------------------------------------------
# Incremental merge pipeline
# ---------------------------------------------------------------------------


@dataclass
class EntityMergeInput:
    """Input for merging a single entity during incremental update."""

    entity_name: str
    existing_type: Optional[str] = None
    existing_description: Optional[str] = None
    existing_source_ids: Optional[List[str]] = None
    new_type: Optional[str] = None
    new_description: Optional[str] = None
    new_source_ids: Optional[List[str]] = None


@dataclass
class EntityMergeOutput:
    """Result of merging a single entity."""

    entity_name: str
    merged_type: str = ""
    merged_description: str = ""
    merged_source_ids: List[str] = field(default_factory=list)
    changed: bool = False  # Whether any field actually changed


@dataclass
class RelationMergeInput:
    """Input for merging a single relation during incremental update."""

    source_entity: str
    target_entity: str
    relation_label: str
    existing_description: Optional[str] = None
    existing_source_ids: Optional[List[str]] = None
    new_description: Optional[str] = None
    new_source_ids: Optional[List[str]] = None


@dataclass
class RelationMergeOutput:
    """Result of merging a single relation."""

    source_entity: str
    target_entity: str
    relation_label: str
    merged_description: str = ""
    merged_source_ids: List[str] = field(default_factory=list)
    changed: bool = False


class IncrementalMergePipeline:
    """Orchestrates entity→relation merge during incremental document insertion.

    This pipeline mirrors LightRAG's two-phase merge:
    - Phase 1: Merge entities (type vote + description merge + source_ids limit)
    - Phase 2: Merge relations (description merge + source_ids limit)

    Usage::

        pipeline = IncrementalMergePipeline()
        entity_out = pipeline.merge_entity(EntityMergeInput(...))
        relation_out = pipeline.merge_relation(RelationMergeInput(...))
    """

    def __init__(
        self,
        source_ids_config: Optional[SourceIdsConfig] = None,
        merge_config: Optional[DescriptionMergerConfig] = None,
        llm_func: Optional[Any] = None,
    ) -> None:
        self._source_ids_mgr = SourceIdsManager(config=source_ids_config)
        self._description_merger = DescriptionMerger(
            config=merge_config, llm_func=llm_func
        )

    def merge_entity(self, input_: EntityMergeInput) -> EntityMergeOutput:
        """Merge a single entity: type vote + description merge + source_ids."""
        # Phase 1a: Vote on entity type
        merged_type = vote_entity_type(
            existing_type=input_.existing_type,
            new_types=[input_.new_type] if input_.new_type else None,
        )

        # Phase 1b: Merge descriptions
        merged_description = merge_entity_descriptions(
            existing_description=input_.existing_description,
            new_descriptions=[input_.new_description] if input_.new_description else None,
            merger=self._description_merger,
        )

        # Phase 1c: Merge source IDs with limit
        merged_source_ids = self._source_ids_mgr.merge(
            existing_ids=input_.existing_source_ids,
            new_ids=input_.new_source_ids,
        )

        # Determine if anything changed
        changed = (
            merged_type != (input_.existing_type or "")
            or merged_description != (input_.existing_description or "")
            or merged_source_ids != list(input_.existing_source_ids or [])
        )

        return EntityMergeOutput(
            entity_name=input_.entity_name,
            merged_type=merged_type,
            merged_description=merged_description,
            merged_source_ids=merged_source_ids,
            changed=changed,
        )

    def merge_relation(self, input_: RelationMergeInput) -> RelationMergeOutput:
        """Merge a single relation: description merge + source_ids."""
        # Phase 2a: Merge descriptions
        merged_description = merge_entity_descriptions(
            existing_description=input_.existing_description,
            new_descriptions=[input_.new_description] if input_.new_description else None,
            merger=self._description_merger,
        )

        # Phase 2b: Merge source IDs with limit
        merged_source_ids = self._source_ids_mgr.merge(
            existing_ids=input_.existing_source_ids,
            new_ids=input_.new_source_ids,
        )

        changed = (
            merged_description != (input_.existing_description or "")
            or merged_source_ids != list(input_.existing_source_ids or [])
        )

        return RelationMergeOutput(
            source_entity=input_.source_entity,
            target_entity=input_.target_entity,
            relation_label=input_.relation_label,
            merged_description=merged_description,
            merged_source_ids=merged_source_ids,
            changed=changed,
        )


# ---------------------------------------------------------------------------
# Delta merge coordinator (KgTableProvider-backed physical isolation)
# ---------------------------------------------------------------------------


class DeltaMergeCoordinator:
    """Persist incremental merge outputs into physically isolated delta tables.

    Generalized from the MS-GraphRAG update pipeline's child-provider
    convention (``KgTableProvider.child("delta")``): every incremental write
    lands in a ``delta_<table>`` vertex label first (isolated from the main
    tables), then :meth:`commit` merges the delta rows into the main tables
    and clears the delta, or :meth:`rollback` discards the pending delta
    without touching the main tables. This gives incremental updates a
    transaction-like boundary without locks: the main tables stay consistent
    until the delta is committed.

    Usage::

        coord = DeltaMergeCoordinator()  # or pass a client / provider
        coord.write_rows("entities", [{"id": "e1", "name": "New"}],
                         schema={"name": "TEXT"})
        coord.has_delta("entities", "e1")          # True
        coord.commit("entities")                    # merged into entities
        coord.rollback()                            # or discard everything
    """

    def __init__(
        self,
        provider: Optional[Any] = None,
        *,
        client: Optional[Any] = None,
        graph_name: Optional[str] = None,
        connection: Optional[Dict[str, Any]] = None,
    ) -> None:
        if provider is not None:
            self._provider = provider
        else:
            from hugegraph_llm.operators.hugegraph_op.kg_table_provider import (
                KgTableProvider,
            )

            self._provider = KgTableProvider(
                client=client, graph_name=graph_name, connection=connection
            )
        self._delta = self._provider.child("delta")
        self._schemas: Dict[str, Optional[Dict[str, str]]] = {}

    @property
    def provider(self) -> Any:
        """The main (non-namespaced) table provider."""
        return self._provider

    @property
    def delta_provider(self) -> Any:
        """The namespaced child provider holding pending delta tables."""
        return self._delta

    def table(self, name: str, schema: Optional[Dict[str, str]] = None) -> Any:
        """Register a table's schema (first use) and return its delta table."""
        self._schemas.setdefault(name, schema)
        return self._delta.table(name, schema=schema)

    def write_rows(
        self,
        table_name: str,
        rows: List[Dict[str, Any]],
        schema: Optional[Dict[str, str]] = None,
    ) -> int:
        """Write rows into the delta table (physically isolated)."""
        return self.table(table_name, schema=schema).upsert_many(rows)

    def list_delta(
        self, table_name: str, schema: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """List pending delta rows for a table."""
        return self.table(table_name, schema=schema).list_rows()

    def has_delta(
        self,
        table_name: str,
        row_id: Any,
        schema: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Check whether a row id exists in the pending delta."""
        return self.table(table_name, schema=schema).has(row_id)

    def commit(self, table_name: Optional[str] = None) -> int:
        """Merge pending delta rows into the main tables, then clear the delta.

        If ``table_name`` is given, only that table is committed; otherwise
        every table registered so far is committed. Returns the number of
        rows merged.
        """
        names = [table_name] if table_name else list(self._schemas.keys())
        merged = 0
        for name in names:
            schema = self._schemas.get(name)
            main_table = self._provider.table(name, schema=schema)
            delta_table = self._delta.table(name, schema=schema)
            for row in delta_table.list_rows():
                main_table.upsert(row)
                merged += 1
            delta_table.clear()
        return merged

    def rollback(self, table_name: Optional[str] = None) -> int:
        """Discard pending delta rows (clear the delta tables).

        If ``table_name`` is given, only that table is rolled back; otherwise
        every registered table is cleared. Returns the number of rows
        discarded.
        """
        names = [table_name] if table_name else list(self._schemas.keys())
        cleared = 0
        for name in names:
            schema = self._schemas.get(name)
            cleared += self._delta.table(name, schema=schema).clear()
        return cleared

    def pending_tables(self) -> List[str]:
        """Tables with a registered schema (pending or already committed)."""
        return list(self._schemas.keys())
