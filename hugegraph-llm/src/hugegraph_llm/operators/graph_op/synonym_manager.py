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

"""Synonym management for knowledge graph quality.

Provides an in-memory synonym registry with disk persistence and
graph-based storage via HugeGraph. Supports:

- Adding synonym groups (canonical term + aliases)
- Expanding queries with synonyms for improved recall
- Persisting to JSON files and/or HugeGraph edges
- Bidirectional lookup: alias -> canonical, canonical -> all aliases

This directly addresses the "semantic mismatch" problem where users
say "actual car model" but the knowledge base uses "physical car model"
(e.g., in metadata retrieval scenarios).

Usage::

    from hugegraph_llm.operators.graph_op.synonym_manager import SynonymManager

    sm = SynonymManager()
    sm.add_synonym("physical car model", ["actual car model", "vehicle type"])
    expanded = sm.expand_query("where is the actual car model field")
    # -> "where is the actual car model physical car model vehicle type field"
"""

import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from hugegraph_llm.config import resource_path

log = logging.getLogger(__name__)

SYNONYM_DATA_FILE = "synonyms.json"

# Edge label for storing synonyms in HugeGraph
SYNONYM_EDGE_LABEL = "SYNONYMOUS_WITH"
# Vertex label for synonym groups (optional, for graph-based storage)
SYNONYM_VERTEX_LABEL = "SynonymGroup"


class SynonymGroup:
    """A group of synonymous terms.

    Attributes:
        group_id: Unique identifier for this synonym group.
        canonical: The canonical/preferred term.
        aliases: Alternative terms that map to the canonical term.
        category: Optional category (e.g., "business_term", "abbreviation").
        metadata: Optional extra metadata.
    """

    def __init__(
        self,
        group_id: str,
        canonical: str,
        aliases: Optional[List[str]] = None,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.group_id = group_id
        self.canonical = canonical
        self.aliases = list(aliases or [])
        self.category = category
        self.metadata = metadata or {}

    @property
    def all_terms(self) -> List[str]:
        """All terms in this group (canonical + aliases)."""
        return [self.canonical] + self.aliases

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "group_id": self.group_id,
            "canonical": self.canonical,
            "aliases": self.aliases,
            "category": self.category,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynonymGroup":
        """Deserialize from dictionary."""
        return cls(
            group_id=data["group_id"],
            canonical=data["canonical"],
            aliases=data.get("aliases", []),
            category=data.get("category", "general"),
            metadata=data.get("metadata", {}),
        )


class SynonymManager:
    """In-memory synonym registry with persistence.

    Supports two storage backends:
    1. **Local JSON file** (default): stores at ``{resource_path}/{graph_name}/synonyms/``
    2. **HugeGraph edges** (optional): stores SYNONYMOUS_WITH edges between term vertices

    The manager maintains a bidirectional index:
    - ``alias -> canonical`` for fast lookup
    - ``canonical -> SynonymGroup`` for full group access

    Usage::

        sm = SynonymManager()
        sm.add_synonym("physical car model", ["actual car model"])
        terms = sm.expand_query("actual car model field")
        sm.save()
    """

    def __init__(self, client: Optional[Any] = None):
        """Initialize synonym manager.

        Args:
            client: Optional HugeGraph client for graph-based storage.
                    If None, only local JSON persistence is used.
        """
        self._client = client
        # group_id -> SynonymGroup
        self._groups: Dict[str, SynonymGroup] = {}
        # Lowercase term -> canonical term (for fast lookup)
        self._term_index: Dict[str, str] = {}
        # Lowercase canonical -> group_id
        self._canonical_index: Dict[str, str] = {}
        # Counter for auto-generating group IDs
        self._next_id = 1

    @property
    def group_count(self) -> int:
        """Number of synonym groups."""
        return len(self._groups)

    def add_synonym(
        self,
        canonical: str,
        aliases: List[str],
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SynonymGroup:
        """Add a synonym group.

        Args:
            canonical: The preferred/canonical term.
            aliases: List of alternative terms.
            category: Optional category label.
            metadata: Optional extra metadata.

        Returns:
            The created SynonymGroup.

        Raises:
            ValueError: If canonical conflicts with an existing alias.
        """
        canonical_lower = canonical.lower().strip()
        if not canonical_lower:
            raise ValueError("Canonical term cannot be empty")

        # Check for conflicts: is canonical already an alias of another group?
        if canonical_lower in self._term_index:
            existing_canonical = self._term_index[canonical_lower]
            if existing_canonical != canonical_lower:
                raise ValueError(
                    f"Canonical '{canonical}' is already an alias of "
                    f"'{existing_canonical}'. Remove it first or merge groups."
                )

        group_id = f"syn_{self._next_id:04d}"
        self._next_id += 1

        group = SynonymGroup(
            group_id=group_id,
            canonical=canonical.strip(),
            aliases=[a.strip() for a in aliases if a.strip()],
            category=category,
            metadata=metadata,
        )

        self._groups[group_id] = group
        self._canonical_index[canonical_lower] = group_id
        self._term_index[canonical_lower] = canonical_lower  # self-referencing

        for alias in group.aliases:
            alias_lower = alias.lower()
            if alias_lower in self._term_index:
                log.warning(
                    "Alias '%s' conflicts with existing term '%s', overwriting",
                    alias,
                    self._term_index[alias_lower],
                )
            self._term_index[alias_lower] = canonical_lower

        log.debug(
            "Added synonym group %s: %s -> %s",
            group_id,
            canonical,
            aliases,
        )
        return group

    def add_alias(self, canonical: str, alias: str) -> bool:
        """Add a single alias to an existing canonical term.

        Args:
            canonical: The canonical term to extend.
            alias: The new alias to add.

        Returns:
            True if added, False if canonical not found.
        """
        canonical_lower = canonical.lower().strip()
        group_id = self._canonical_index.get(canonical_lower)
        if not group_id:
            return False

        group = self._groups[group_id]
        alias_lower = alias.lower().strip()
        if alias_lower not in self._term_index:
            group.aliases.append(alias.strip())
            self._term_index[alias_lower] = canonical_lower
            return True
        return False

    def remove_group(self, canonical: str) -> bool:
        """Remove a synonym group by canonical term.

        Args:
            canonical: The canonical term of the group to remove.

        Returns:
            True if removed, False if not found.
        """
        canonical_lower = canonical.lower().strip()
        group_id = self._canonical_index.pop(canonical_lower, None)
        if not group_id:
            return False

        group = self._groups.pop(group_id, None)
        if group:
            # Remove all term mappings
            for term in group.all_terms:
                self._term_index.pop(term.lower(), None)
        return True

    def lookup(self, term: str) -> Optional[str]:
        """Look up the canonical term for a given term.

        Args:
            term: Any term (canonical or alias).

        Returns:
            The canonical term, or the term itself if not found.
        """
        canonical_lower = self._term_index.get(term.lower().strip())
        if canonical_lower:
            # Find the actual canonical (non-lowercased) from the group
            group_id = self._canonical_index.get(canonical_lower)
            if group_id and group_id in self._groups:
                return self._groups[group_id].canonical
            return canonical_lower
        return None

    def get_group(self, canonical: str) -> Optional[SynonymGroup]:
        """Get the full synonym group for a canonical term.

        Args:
            canonical: The canonical term.

        Returns:
            SynonymGroup or None if not found.
        """
        group_id = self._canonical_index.get(canonical.lower().strip())
        if group_id and group_id in self._groups:
            return self._groups[group_id]
        return None

    def expand_query(self, query: str) -> str:
        """Expand a query with synonyms for improved recall.

        Uses jieba to tokenize Chinese text, then checks each token
        and bigram against the synonym registry. Also checks the
        full query string as a whole for exact synonym matches.

        Args:
            query: Original query text.

        Returns:
            Expanded query string with synonyms appended.
        """
        import re

        # Tokenize: split on non-word boundaries but keep Chinese chars grouped
        # Use jieba for Chinese-aware tokenization
        import jieba as _jieba

        raw_tokens = _jieba.lcut(query)
        # Keep only meaningful tokens (alphanumeric or Chinese)
        tokens = [t for t in raw_tokens if re.match(r"^[\w\u4e00-\u9fff]+$", t)]

        added_synonyms: Set[str] = set()

        # Check single tokens
        for token in tokens:
            self._collect_synonyms(token, tokens, added_synonyms)

        # Check bigrams for multi-character Chinese terms
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + tokens[i + 1]
            if any("\u4e00" <= c <= "\u9fff" for c in bigram):
                self._collect_synonyms(bigram, tokens, added_synonyms)

        # Check the full query as a whole (for exact match)
        self._collect_synonyms(query.strip(), tokens, added_synonyms)

        if not added_synonyms:
            return query

        return query + " " + " ".join(sorted(added_synonyms))

    def _collect_synonyms(
        self,
        text: str,
        original_tokens: List[str],
        added: Set[str],
    ) -> None:
        """Collect all synonyms for a text term."""
        canonical = self.lookup(text)
        if canonical:
            group = self.get_group(canonical)
            if group:
                for term in group.all_terms:
                    added.add(term)

    def expand_tokens(self, tokens: List[str]) -> List[str]:
        """Expand a token list with synonyms.

        Args:
            tokens: Original token list.

        Returns:
            Expanded token list with synonyms included.
        """
        expanded = list(tokens)
        seen = set(t.lower() for t in tokens)

        for token in tokens:
            canonical = self.lookup(token)
            if canonical:
                group = self.get_group(canonical)
                if group:
                    for term in group.all_terms:
                        if term.lower() not in seen:
                            expanded.append(term)
                            seen.add(term.lower())

        return expanded

    def save(self, *name: Optional[str]) -> None:
        """Save synonym data to disk.

        Args:
            *name: Path components under resource_path. If empty,
                   defaults to ``(huge_settings.graph_name, "synonyms")``.
        """
        from hugegraph_llm.config import huge_settings

        path_parts = name if name else (huge_settings.graph_name, "synonyms")
        dir_path = os.path.join(resource_path, *path_parts)
        os.makedirs(dir_path, exist_ok=True)

        data = {
            "next_id": self._next_id,
            "groups": {
                gid: g.to_dict() for gid, g in self._groups.items()
            },
        }

        filepath = os.path.join(dir_path, SYNONYM_DATA_FILE)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        log.debug(
            "Saved %d synonym groups to %s", self.group_count, filepath
        )

    @classmethod
    def from_saved(cls, *name: Optional[str]) -> "SynonymManager":
        """Load synonym data from disk.

        Args:
            *name: Path components. If empty, defaults to
                   ``(huge_settings.graph_name, "synonyms")``.

        Returns:
            Loaded SynonymManager, or empty instance if no saved data.
        """
        from hugegraph_llm.config import huge_settings

        path_parts = name if name else (huge_settings.graph_name, "synonyms")
        dir_path = os.path.join(resource_path, *path_parts)
        filepath = os.path.join(dir_path, SYNONYM_DATA_FILE)

        if not os.path.exists(filepath):
            log.debug("No saved synonyms at %s", filepath)
            return cls()

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            instance = cls()
            instance._next_id = data.get("next_id", 1)

            for gid, gdata in data.get("groups", {}).items():
                group = SynonymGroup.from_dict(gdata)
                instance._groups[gid] = group
                instance._canonical_index[group.canonical.lower()] = gid
                instance._term_index[group.canonical.lower()] = group.canonical.lower()
                for alias in group.aliases:
                    instance._term_index[alias.lower()] = group.canonical.lower()

            log.debug(
                "Loaded %d synonym groups from %s",
                instance.group_count,
                filepath,
            )
            return instance
        except Exception as e:
            log.warning("Failed to load synonyms from %s: %s", filepath, e)
            return cls()

    def to_graph_edges(self) -> List[Dict[str, Any]]:
        """Convert synonym groups to graph edge format.

        Returns a list of edge dicts suitable for Commit2HugeGraph::

            [
                {
                    "source": "physical car model",
                    "target": "actual car model",
                    "label": "SYNONYMOUS_WITH",
                    "properties": {"group_id": "syn_0001"}
                },
                ...
            ]
        """
        edges = []
        for group in self._groups.values():
            for alias in group.aliases:
                edges.append(
                    {
                        "source": group.canonical,
                        "target": alias,
                        "label": SYNONYM_EDGE_LABEL,
                        "properties": {
                            "group_id": group.group_id,
                            "category": group.category,
                        },
                    }
                )
        return edges

    def import_from_edges(
        self,
        edges: List[Dict[str, Any]],
    ) -> int:
        """Import synonym relationships from graph edges.

        Args:
            edges: List of edge dicts with ``source``, ``target``,
                   and optionally ``properties.category``.

        Returns:
            Number of synonym groups created.
        """
        groups: Dict[str, List[str]] = defaultdict(list)
        categories: Dict[str, str] = {}

        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            label = edge.get("label", "")
            if label != SYNONYM_EDGE_LABEL:
                continue
            if source and target:
                groups[source].append(target)
                categories[source] = edge.get("properties", {}).get(
                    "category", "general"
                )

        count = 0
        for canonical, aliases in groups.items():
            try:
                self.add_synonym(
                    canonical=canonical,
                    aliases=aliases,
                    category=categories.get(canonical, "general"),
                )
                count += 1
            except ValueError:
                log.warning("Conflict importing synonym group for '%s'", canonical)

        return count
